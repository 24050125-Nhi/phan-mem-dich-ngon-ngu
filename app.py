import csv
import io
import json
import os
import re
from difflib import SequenceMatcher
from flask import Flask, request, jsonify, render_template_string, Response

app = Flask(__name__)

DICT_FILE = "dictionary.json"
PHRASES_FILE = "phrases.json"

DEFAULT_DICT = {
    "khảm": {"vi": "hỏi", "region": "chung"},
    "pây": {"vi": "đi", "region": "chung"},
    "mạ": {"vi": "về", "region": "chung"},
    "kin": {"vi": "ăn", "region": "chung"},
    "nặm": {"vi": "nước", "region": "chung"},
    "hưn": {"vi": "nhà", "region": "chung"},
    "khau": {"vi": "gạo / cơm", "region": "chung"},
    "kin khau": {"vi": "ăn cơm", "region": "chung"},
    "slương": {"vi": "lên", "region": "chung"},
    "lồng": {"vi": "xuống", "region": "chung"},
    "bjooc": {"vi": "hoa", "region": "chung"},
    "pò": {"vi": "bố / cha", "region": "chung"},
    "mẹ": {"vi": "mẹ", "region": "chung"},
    "lừm": {"vi": "quên", "region": "chung"},
    "rú": {"vi": "biết", "region": "chung"},
    "tồng": {"vi": "rừng", "region": "chung"},
}

DEFAULT_PHRASES = [
    {"tay": "pây hưn", "vi": "đi về nhà", "region": "chung"},
    {"tay": "kin khau slương", "vi": "ăn cơm chưa / ăn cơm lên", "region": "chung"},
    {"tay": "pây tồng mạ", "vi": "đi rừng về", "region": "chung"},
]

WORD_RE = re.compile(r"^\w+$", flags=re.UNICODE)
TOKEN_RE = re.compile(r"\w+|[^\w\s]+|\s+", flags=re.UNICODE)
PUNCT_SPACE_RE = re.compile(r"\s+([.,!?;:])")
MULTI_SPACE_RE = re.compile(r"\s{2,}")

# Cache trong bộ nhớ kèm thời điểm sửa file (mtime)
_DICT_CACHE = {
    "mtime": None,
    "data": {},
    "lookup": {},   # tuple(words) -> entry dict
    "max_len": 1,
}

_PHRASES_CACHE = {
    "mtime": None,
    "raw": [],
    "prepared": [],  # list of (tay_orig, tay_norm, len_norm, vi, region)
}


def normalize_entry(raw_val) -> dict:
    if isinstance(raw_val, dict):
        return {
            "vi": str(raw_val.get("vi", "")).strip(),
            "region": str(raw_val.get("region", "chung")).strip() or "chung",
        }
    return {"vi": str(raw_val).strip(), "region": "chung"}


def save_dictionary(data: dict) -> None:
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # Làm mới cache ngay sau khi ghi
    get_dictionary_bundle(force_reload=True)


def save_phrases(data: list) -> None:
    with open(PHRASES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    get_phrases_bundle(force_reload=True)


def get_dictionary_bundle(force_reload: bool = False) -> dict:
    if not os.path.exists(DICT_FILE):
        save_dictionary(DEFAULT_DICT)

    mtime = os.path.getmtime(DICT_FILE)
    if (
        not force_reload
        and _DICT_CACHE["mtime"] == mtime
        and _DICT_CACHE["lookup"] is not None
    ):
        return _DICT_CACHE

    with open(DICT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned_data = {}
    lookup = {}
    max_len = 1

    for k, v in raw_data.items():
        if not k or not str(k).strip():
            continue
        norm_key = " ".join(str(k).strip().lower().split())
        entry = normalize_entry(v)
        if not entry["vi"]:
            continue
        cleaned_data[norm_key] = entry
        parts = tuple(norm_key.split())
        lookup[parts] = entry
        if len(parts) > max_len:
            max_len = len(parts)

    _DICT_CACHE["mtime"] = mtime
    _DICT_CACHE["data"] = cleaned_data
    _DICT_CACHE["lookup"] = lookup
    _DICT_CACHE["max_len"] = min(max_len, 8)
    return _DICT_CACHE


def get_phrases_bundle(force_reload: bool = False) -> dict:
    if not os.path.exists(PHRASES_FILE):
        save_phrases(DEFAULT_PHRASES)

    mtime = os.path.getmtime(PHRASES_FILE)
    if not force_reload and _PHRASES_CACHE["mtime"] == mtime:
        return _PHRASES_CACHE

    with open(PHRASES_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_list = raw_data if isinstance(raw_data, list) else []
    prepared = []
    for p in raw_list:
        tay_orig = str(p.get("tay", "")).strip()
        tay_norm = " ".join(tay_orig.lower().split())
        if not tay_norm:
            continue
        prepared.append((
            tay_orig,
            tay_norm,
            len(tay_norm),
            str(p.get("vi", "")).strip(),
            str(p.get("region", "chung")).strip() or "chung",
        ))

    _PHRASES_CACHE["mtime"] = mtime
    _PHRASES_CACHE["raw"] = raw_list
    _PHRASES_CACHE["prepared"] = prepared
    return _PHRASES_CACHE


def translate_by_dictionary(text: str, lookup: dict, max_ngram: int):
    """
    Khớp tham lam cụm từ dài nhất trước (tối đa max_ngram từ), sau đó mới tới từ đơn.
    Dùng tuple từ vựng để tra O(1) thay vì ghép chuỗi nhiều lần.
    """
    tokens = TOKEN_RE.findall(text.lower())
    word_indices = [i for i, tok in enumerate(tokens) if WORD_RE.match(tok)]
    words_only = [tokens[i] for i in word_indices]
    num_words = len(words_only)

    if num_words == 0:
        return text.strip(), {}

    matched = {}
    used_word_pos = set()
    phrase_at_pos = {}

    upper_n = min(max_ngram, num_words)
    for n in range(upper_n, 0, -1):
        for start in range(0, num_words - n + 1):
            # Bỏ qua nhanh nếu bất kỳ vị trí từ nào đã được dùng bởi cụm dài hơn
            has_overlap = False
            for offset in range(n):
                if (start + offset) in used_word_pos:
                    has_overlap = True
                    break
            if has_overlap:
                continue

            candidate_tuple = tuple(words_only[start : start + n])
            entry = lookup.get(candidate_tuple)
            if entry is not None:
                phrase_key = " ".join(candidate_tuple)
                matched[phrase_key] = entry
                phrase_at_pos[start] = (n, entry["vi"])
                for offset in range(n):
                    used_word_pos.add(start + offset)

    word_pos_map = {token_idx: pos for pos, token_idx in enumerate(word_indices)}
    output_parts = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        pos = word_pos_map.get(i)
        if pos is not None:
            phrase_info = phrase_at_pos.get(pos)
            if phrase_info is not None:
                n_words, vi_meaning = phrase_info
                output_parts.append(vi_meaning)
                matched_count = 0
                while i < len(tokens) and matched_count < n_words:
                    if WORD_RE.match(tokens[i]):
                        matched_count += 1
                    i += 1
                continue
            output_parts.append(f"[{tok}]")
        else:
            output_parts.append(tok)
        i += 1

    cleaned = PUNCT_SPACE_RE.sub(r"\1", "".join(output_parts))
    cleaned = MULTI_SPACE_RE.sub(" ", cleaned).strip()
    return cleaned, matched


def find_similar_phrases(text: str, prepared_phrases: list, limit: int = 3):
    if not prepared_phrases:
        return []
    q = " ".join(text.lower().split())
    q_len = len(q)
    if q_len == 0:
        return []

    scored = []
    for tay_orig, tay_norm, tay_len, vi, region in prepared_phrases:
        # Lọc nhanh: nếu độ dài chênh quá lớn thì tỷ lệ tương đồng tối đa không thể đạt >= 0.45
        if min(q_len, tay_len) * 2.2 < max(q_len, tay_len):
            continue
        ratio = SequenceMatcher(None, q, tay_norm).ratio()
        if ratio >= 0.45:
            scored.append({
                "tay": tay_orig,
                "vi": vi,
                "region": region,
                "score": round(ratio, 2),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Dịch Tiếng Tày – Tiếng Việt </title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 940px; margin: 28px auto; padding: 0 16px; background: #f6f7f9; color: #222; }
    h1 { font-size: 24px; margin-bottom: 6px; }
    .sub { color: #555; margin-bottom: 20px; font-size: 14px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 680px) { .grid { grid-template-columns: 1fr; } }
    textarea { width: 100%; height: 140px; padding: 12px; font-size: 15px; border: 1px solid #ccc; border-radius: 8px; box-sizing: border-box; }
    .box { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; min-height: 140px; font-size: 15px; white-space: pre-wrap; }
    button { padding: 9px 15px; font-size: 14px; border: none; border-radius: 7px; background: #1a73e8; color: #fff; cursor: pointer; }
    button:hover:not(:disabled) { background: #1558b0; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    button.secondary { background: #4b5563; }
    button.secondary:hover:not(:disabled) { background: #374151; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-top: 20px; }
    input { padding: 8px 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; }
    .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }
    .meta { margin-top: 10px; font-size: 13px; color: #444; }
    .similar { margin-top: 10px; background: #f9fafb; border-left: 3px solid #1a73e8; padding: 8px 12px; font-size: 13px; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }
    th, td { border-bottom: 1px solid #eee; padding: 8px; text-align: left; }
    th { background: #f9fafb; }
  </style>
</head>
<body>
  <h1>Phần mềm dịch Tiếng Tày → Tiếng Việt </h1>
  <div class="sub">Cache từ điển trong bộ nhớ + tra cụm từ nhanh. Nhấn <strong>Ctrl + Enter</strong> trong ô nhập để dịch nhanh.</div>

  <div class="grid">
    <div>
      <label><strong>Tiếng Tày</strong></label>
      <textarea id="tayInput" placeholder="Ví dụ: pây hưn, kin khau slương..."></textarea>
      <div class="row">
        <button id="translateBtn" onclick="translateText()">Dịch từ điển</button>
        <span id="latencyBadge" style="font-size:12px; color:#666;"></span>
      </div>
    </div>
    <div>
      <label><strong>Tiếng Việt</strong></label>
      <div id="viOutput" class="box">Kết quả dịch sẽ hiển thị tại đây...</div>
      <div id="matchedInfo" class="meta"></div>
      <div id="similarBox" class="similar" style="display:none;"></div>
    </div>
  </div>

  <div class="card">
    <h3>1. Thêm / cập nhật từ hoặc cụm từ vào từ điển</h3>
    <div class="row">
      <input id="newTay" placeholder="Từ/cụm từ tiếng Tày" />
      <input id="newVi" placeholder="Nghĩa tiếng Việt" />
      <input id="newRegion" placeholder="Vùng/ngữ cảnh" value="chung" />
      <button onclick="addWord()">Lưu từ điển</button>
      <span id="saveMsg" style="font-size:13px; color:green;"></span>
    </div>
  </div>

  <div class="card">
    <h3>2. Thêm câu mẫu song ngữ (để đối chiếu câu tương tự)</h3>
    <div class="row">
      <input id="phraseTay" placeholder="Câu tiếng Tày" />
      <input id="phraseVi" placeholder="Bản dịch tiếng Việt chuẩn" />
      <button onclick="addPhrase()">Lưu câu mẫu</button>
      <span id="phraseMsg" style="font-size:13px; color:green;"></span>
    </div>
  </div>

  <div class="card">
    <h3>3. Tìm kiếm & quản lý CSV</h3>
    <div class="row">
      <input id="searchQuery" placeholder="Tìm từ Tày hoặc nghĩa Việt..." oninput="searchDictionary()" />
      <button class="secondary" onclick="exportCsv()">Xuất CSV</button>
      <input type="file" id="csvFile" accept=".csv" />
      <button class="secondary" onclick="importCsv()">Nhập CSV</button>
      <span id="csvMsg" style="font-size:13px; color:#1a73e8;"></span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Tiếng Tày / Cụm từ</th>
          <th>Tiếng Việt</th>
          <th>Vùng / Ngữ cảnh</th>
        </tr>
      </thead>
      <tbody id="dictTableBody"></tbody>
    </table>
  </div>

  <script>
    let isTranslating = false;

    document.getElementById("tayInput").addEventListener("keydown", function(e) {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        translateText();
      }
    });

    async function translateText() {
      if (isTranslating) return;
      const text = document.getElementById("tayInput").value.trim();
      if (!text) return;

      isTranslating = true;
      const btn = document.getElementById("translateBtn");
      btn.disabled = true;
      const t0 = performance.now();

      try {
        const res = await fetch("/translate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });
        const data = await res.json();
        const elapsed = Math.round(performance.now() - t0);
        document.getElementById("latencyBadge").innerText = `Xong trong ${elapsed} ms`;
        document.getElementById("viOutput").innerText = data.translation || data.error || "Không có kết quả.";

        if (data.matched && Object.keys(data.matched).length > 0) {
          const pairs = Object.entries(data.matched).map(([k, v]) => `${k}: ${v.vi}`).join(" | ");
          document.getElementById("matchedInfo").innerText = "Đã khớp: " + pairs;
        } else {
          document.getElementById("matchedInfo").innerText = "Không khớp mục nào trong từ điển.";
        }

        const simBox = document.getElementById("similarBox");
        if (data.similar_phrases && data.similar_phrases.length > 0) {
          simBox.style.display = "block";
          simBox.innerHTML = "<strong>Câu mẫu gần giống:</strong><br>" + data.similar_phrases.map(
            p => `• <em>${p.tay}</em> → <strong>${p.vi}</strong> (độ khớp ${Math.round(p.score * 100)}%)`
          ).join("<br>");
        } else {
          simBox.style.display = "none";
        }
      } finally {
        isTranslating = false;
        btn.disabled = false;
      }
    }

    async function addWord() {
      const tay = document.getElementById("newTay").value.trim();
      const vi = document.getElementById("newVi").value.trim();
      const region = document.getElementById("newRegion").value.trim() || "chung";
      if (!tay || !vi) return;
      const res = await fetch("/dictionary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tay, vi, region })
      });
      const data = await res.json();
      document.getElementById("saveMsg").innerText = data.message || data.error || "";
      document.getElementById("newTay").value = "";
      document.getElementById("newVi").value = "";
      searchDictionary();
    }

    async function addPhrase() {
      const tay = document.getElementById("phraseTay").value.trim();
      const vi = document.getElementById("phraseVi").value.trim();
      if (!tay || !vi) return;
      const res = await fetch("/phrases", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tay, vi })
      });
      const data = await res.json();
      document.getElementById("phraseMsg").innerText = data.message || data.error || "";
      document.getElementById("phraseTay").value = "";
      document.getElementById("phraseVi").value = "";
    }

    async function searchDictionary() {
      const q = encodeURIComponent(document.getElementById("searchQuery").value.trim());
      const res = await fetch(`/dictionary/search?q=${q}`);
      const rows = await res.json();
      const tbody = document.getElementById("dictTableBody");
      tbody.innerHTML = rows.slice(0, 50).map(r =>
        `<tr><td><strong>${r.tay}</strong></td><td>${r.vi}</td><td>${r.region}</td></tr>`
      ).join("") || `<tr><td colspan="3">Không tìm thấy từ phù hợp.</td></tr>`;
    }

    function exportCsv() {
      window.location.href = "/dictionary/export";
    }

    async function importCsv() {
      const fileInput = document.getElementById("csvFile");
      if (!fileInput.files.length) return;
      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      const res = await fetch("/dictionary/import", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      document.getElementById("csvMsg").innerText = data.message || data.error || "";
      searchDictionary();
    }

    searchDictionary();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/translate", methods=["POST"])
def translate():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Vui lòng nhập nội dung tiếng Tày."}), 400

    dict_bundle = get_dictionary_bundle()
    phrases_bundle = get_phrases_bundle()

    translation, matched = translate_by_dictionary(
        text, dict_bundle["lookup"], dict_bundle["max_len"]
    )
    similar = find_similar_phrases(text, phrases_bundle["prepared"])

    return jsonify({
        "translation": translation,
        "matched": matched,
        "similar_phrases": similar,
    })


@app.route("/dictionary", methods=["POST"])
def add_dictionary_entry():
    payload = request.get_json(silent=True) or {}
    tay = " ".join((payload.get("tay") or "").strip().lower().split())
    vi = (payload.get("vi") or "").strip()
    region = (payload.get("region") or "chung").strip() or "chung"
    if not tay or not vi:
        return jsonify({"error": "Cần nhập đủ từ tiếng Tày và nghĩa tiếng Việt."}), 400

    dict_bundle = get_dictionary_bundle()
    updated = dict(dict_bundle["data"])
    updated[tay] = {"vi": vi, "region": region}
    save_dictionary(updated)
    return jsonify({"message": f"Đã lưu '{tay}' → '{vi}' ({region})."})


@app.route("/phrases", methods=["POST"])
def add_phrase_entry():
    payload = request.get_json(silent=True) or {}
    tay = " ".join((payload.get("tay") or "").strip().lower().split())
    vi = (payload.get("vi") or "").strip()
    if not tay or not vi:
        return jsonify({"error": "Cần nhập đủ câu tiếng Tày và bản dịch tiếng Việt."}), 400

    phrases_bundle = get_phrases_bundle()
    updated = list(phrases_bundle["raw"])
    updated.append({"tay": tay, "vi": vi, "region": "chung"})
    save_phrases(updated)
    return jsonify({"message": f"Đã lưu câu mẫu '{tay}' → '{vi}'."})


@app.route("/dictionary/search", methods=["GET"])
def search_dictionary():
    q = (request.args.get("q") or "").strip().lower()
    dict_bundle = get_dictionary_bundle()
    results = []
    for tay, info in dict_bundle["data"].items():
        vi = info["vi"]
        region = info["region"]
        if not q or q in tay or q in vi.lower() or q in region.lower():
            results.append({"tay": tay, "vi": vi, "region": region})
    return jsonify(results)


@app.route("/dictionary/export", methods=["GET"])
def export_dictionary():
    dict_bundle = get_dictionary_bundle()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["tay", "vi", "region"])
    for tay, info in dict_bundle["data"].items():
        writer.writerow([tay, info["vi"], info["region"]])

    csv_bytes = ("\ufeff" + output.getvalue()).encode("utf-8")
    return Response(
        csv_bytes,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=tu_dien_tay_viet.csv"},
    )


@app.route("/dictionary/import", methods=["POST"])
def import_dictionary():
    if "file" not in request.files:
        return jsonify({"error": "Không tìm thấy file CSV."}), 400
    file = request.files["file"]
    content = file.stream.read().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(content))

    dict_bundle = get_dictionary_bundle()
    updated = dict(dict_bundle["data"])
    added = 0
    for row in reader:
        tay = " ".join((row.get("tay") or "").strip().lower().split())
        vi = (row.get("vi") or "").strip()
        region = (row.get("region") or "chung").strip() or "chung"
        if tay and vi:
            updated[tay] = {"vi": vi, "region": region}
            added += 1

    save_dictionary(updated)
    return jsonify({"message": f"Đã nhập/cập nhật {added} từ từ file CSV."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    get_dictionary_bundle(force_reload=True)
    get_phrases_bundle(force_reload=True)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)