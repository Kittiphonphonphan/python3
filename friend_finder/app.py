from flask import Flask, render_template_string, request
from mistralai import Mistral
import dotenv, os, json, logging
import numpy as np
from pathlib import Path
from datetime import datetime

dotenv.load_dotenv()

DATA_PATH = Path("data/messages.json")
LOG_PATH = Path("logs/app.log")

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

app = Flask(__name__)

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def cosine_similarity(emb1, emb2):
    emb1 = np.array(emb1, dtype=float)
    emb2 = np.array(emb2, dtype=float)
    denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    if denom == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / denom)

def get_embedding(text: str) -> list[float]:
    res = client.embeddings.create(model="mistral-embed", inputs=[text])
    return res.data[0].embedding

def load_messages():
    if not DATA_PATH.exists():
        return []
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))

def save_messages(messages):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")

def llm_filter_relevant(user_message: str, top3: list[dict]) -> list[dict]:
    """
    ให้ LLM ตัดสินว่าใน top-3 ข้อความไหน 'เกี่ยวจริง' และควรแนะนำเป็นเพื่อน
    คืนเป็น list ของ candidates ที่เลือก
    """
    if not top3:
        return []

    candidates_text = "\n".join(
        [f"{i+1}. nickname={c['nickname']} | message={c['message']} | score={c['score']:.4f}"
         for i, c in enumerate(top3)]
    )

    prompt = f"""
You are helping match people with similar thoughts.
User message:
{user_message}

Top-3 candidates:
{candidates_text}

Task:
Select which candidates are truly relevant for "friend for the moment".
Return ONLY JSON in this format:
{{
  "selected": [1, 2]
}}
Where numbers refer to candidate indices 1..3.
If none are relevant, return {{"selected":[]}}.
"""

    resp = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    content = resp.choices[0].message.content.strip()

    try:
        data = json.loads(content)
        selected = data.get("selected", [])
        selected = [i for i in selected if i in [1, 2, 3]]
        return [top3[i-1] for i in selected]
    except Exception:
        return top3[:1]

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Find a friend for the moment</title>
  <style>
    body { font-family: Arial; max-width: 900px; margin: 24px auto; padding: 0 12px; }
    input, textarea, button { width: 100%; padding: 10px; margin-top: 6px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 12px; margin-top: 12px; }
    .muted { color: #666; font-size: 14px; }
  </style>
</head>
<body>
  <h2>Find a friend for the moment</h2>
  <p class="muted">Enter nickname + message. We will find top-3 by cosine similarity, then use LLM to pick truly relevant ones.</p>

  <form method="post">
    <label>Nickname</label>
    <input name="nickname" required>

    <label style="margin-top:10px;">Message</label>
    <textarea name="message" rows="4" required></textarea>

    <button style="margin-top:10px;" type="submit">Find friend</button>
  </form>

  {% if top3 %}
    <div class="card">
      <h3>Top-3 by cosine similarity</h3>
      <ol>
        {% for c in top3 %}
          <li><b>{{c.nickname}}</b> ({{"%.4f"|format(c.score)}})<br>{{c.message}}</li>
        {% endfor %}
      </ol>
    </div>
  {% endif %}

  {% if recs %}
    <div class="card">
      <h3>Friend recommendation</h3>
      <ul>
        {% for r in recs %}
          <li><b>{{r.nickname}}</b><br>{{r.message}}</li>
        {% endfor %}
      </ul>
    </div>
  {% elif top3 %}
    <div class="card">
      <h3>Friend recommendation</h3>
      <p class="muted">No strong recommendation from LLM.</p>
    </div>
  {% endif %}
</body>
</html>
"""
@app.route("/", methods=["GET", "POST"])
def index():
    top3 = []
    recs = []

    if request.method == "POST":
        nickname = request.form.get("nickname", "").strip()
        message = request.form.get("message", "").strip()

        if nickname and message:
            messages = load_messages()

            new_emb = get_embedding(message)

            scored = []
            for item in messages:
                score = cosine_similarity(new_emb, item["embedding"])
                scored.append({
                    "nickname": item["nickname"],
                    "message": item["message"],
                    "score": score
                })

            scored.sort(key=lambda x: x["score"], reverse=True)
            top3 = scored[:3]

            # LOG top-3
            logging.info(
                f"TOP3 for '{nickname}': " +
                ", ".join([f"{c['nickname']}:{c['score']:.4f}" for c in top3])
            )

            recs = llm_filter_relevant(message, top3)

            messages.append({
                "nickname": nickname,
                "message": message,
                "embedding": new_emb,
                "created_at": datetime.utcnow().isoformat()
            })
            save_messages(messages)

            logging.info(f"ADD message from '{nickname}': {message}")

    return render_template_string(HTML, top3=top3, recs=recs)

if __name__ == "__main__":
    app.run(debug=True)
