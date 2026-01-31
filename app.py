import gradio as gr
import subprocess
from pathlib import Path
import pandas as pd
import time
import html

DATA_FILES = [
    "anime.csv", "brand.csv", "tieup.csv", "tieup_sources.csv",
    "anime_with_clusters.csv", "brand_with_clusters.csv",
    "class_pair_affinity_long.csv",
    "anime_kmeans.joblib", "brand_kmeans.joblib",
]

RECOMMEND_JSON = Path("recommend_out.json")


def run_cmd_list(cmd_list):
    p = subprocess.run(cmd_list, capture_output=True, text=True)
    out = ""
    if p.stdout:
        out += p.stdout
    if p.stderr:
        out += "\n[stderr]\n" + p.stderr
    return out


def list_status():
    lines = []
    for f in DATA_FILES:
        p = Path(f)
        if p.exists():
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
            lines.append(f"✅ {f}  (updated: {ts})")
        else:
            lines.append(f"❌ {f}  (missing)")
    return "\n".join(lines)


def ui_run_clustering(k_anime, k_brand, auto_k):
    cmd = [
        "python", "clustering.py",
        "--k_anime", str(int(k_anime)),
        "--k_brand", str(int(k_brand)),
    ]
    if auto_k:
        cmd.append("--auto_k")

    log = run_cmd_list(cmd)
    return log, list_status()


def ui_run_tieup_matrix(weight_promo):
    cmd = [
        "python", "tieup_matrix.py",
        "--weight_promo", str(float(weight_promo)),
    ]
    log = run_cmd_list(cmd)
    return log, list_status()


def _cards_html_from_df(df: pd.DataFrame, top_n: int = 5) -> str:
    if df is None or df.empty:
        return "<div style='padding:12px; border-radius:12px; background:#fff3cd; border:1px solid #ffeeba;'>結果がありません。</div>"

    cols = [c for c in ["brand_name", "category", "personality_tags", "score", "affinity", "sim", "reason"] if c in df.columns]
    d = df[cols].head(top_n).copy()

    def safe(x):
        return html.escape("" if pd.isna(x) else str(x))

    blocks = []
    for i, row in d.reset_index(drop=True).iterrows():
        name = safe(row.get("brand_name", ""))
        category = safe(row.get("category", ""))
        persona = safe(row.get("personality_tags", ""))
        reason = safe(row.get("reason", ""))

        def fmt(v):
            try:
                return f"{float(v):.3f}"
            except Exception:
                return safe(v)

        score_s = fmt(row.get("score", ""))
        affinity_s = fmt(row.get("affinity", ""))
        sim_s = fmt(row.get("sim", ""))

        blocks.append(f"""
        <div style="
            border: 2px solid #111;
            border-radius: 16px;
            padding: 14px 14px;
            margin: 10px 0;
            background: #ffffff;
            box-shadow: 0 10px 22px rgba(0,0,0,0.10);
        ">
          <div style="display:flex; align-items:center; gap:10px;">
            <div style="
                width:34px; height:34px; border-radius:10px;
                background:#111; color:#fff;
                display:flex; align-items:center; justify-content:center;
                font-weight:900;
            ">{i+1}</div>

            <div style="font-size: 20px; font-weight: 900; line-height:1.2;">
              {name}
            </div>
          </div>

          <div style="margin-top:8px; display:flex; flex-wrap:wrap; gap:8px;">
            {"<span style='padding:6px 10px; border-radius:999px; background:#f3f4f6; border:1px solid #e5e7eb; font-weight:700;'>カテゴリ: " + category + "</span>" if category else ""}
            {"<span style='padding:6px 10px; border-radius:999px; background:#f3f4f6; border:1px solid #e5e7eb;'>イメージ: " + persona + "</span>" if persona else ""}
          </div>

          <div style="margin-top:10px; display:flex; flex-wrap:wrap; gap:10px;">
            <div style="padding:8px 10px; border-radius:12px; background:#111; color:#fff; font-weight:900;">
              score {score_s}
            </div>
            <div style="padding:8px 10px; border-radius:12px; background:#fff; border:1px solid #e5e7eb;">
              相性(affinity) {affinity_s}
            </div>
            <div style="padding:8px 10px; border-radius:12px; background:#fff; border:1px solid #e5e7eb;">
              近さ(sim) {sim_s}
            </div>
          </div>

          <div style="margin-top:10px; padding:10px 12px; border-radius:12px; background:#f9fafb; border:1px solid #e5e7eb;">
            <div style="font-weight:900; margin-bottom:4px;">理由</div>
            <div style="line-height:1.55;">{reason}</div>
          </div>
        </div>
        """)

    return "<div>" + "\n".join(blocks) + "</div>"


def ui_run_recommend(
    title, main_genre, tone_tags, synopsis,
    top_k, top_brand_clusters, w_affinity, w_sim, max_sources
):
    try:
        if RECOMMEND_JSON.exists():
            RECOMMEND_JSON.unlink()
    except Exception:
        pass

    cmd = [
        "python", "recommend.py",
        "--title", str(title),
        "--main_genre", str(main_genre),
        "--tone_tags", str(tone_tags),
        "--synopsis", str(synopsis),
        "--top_k", str(int(top_k)),
        "--top_brand_clusters", str(int(top_brand_clusters)),
        "--w_affinity", str(float(w_affinity)),
        "--w_sim", str(float(w_sim)),
        "--max_sources", str(int(max_sources)),
        "--out_json", str(RECOMMEND_JSON),
    ]

    log = run_cmd_list(cmd)

    df_display = pd.DataFrame()
    group_title = ""
    group_detail = ""
    cluster_note = ""

    if RECOMMEND_JSON.exists():
        try:
            df = pd.read_json(RECOMMEND_JSON)

            # group_* を優先して読む
            if "group_title" in df.columns:
                s = df["group_title"].dropna().astype(str)
                if len(s) > 0 and s.iloc[0].strip():
                    group_title = s.iloc[0].strip()

            if "group_detail" in df.columns:
                s = df["group_detail"].dropna().astype(str)
                if len(s) > 0 and s.iloc[0].strip():
                    group_detail = s.iloc[0].strip()

            if "cluster_note" in df.columns:
                s = df["cluster_note"].dropna().astype(str)
                if len(s) > 0 and s.iloc[0].strip():
                    cluster_note = s.iloc[0].strip()

            # 互換: group_desc しか無い場合
            if (not group_title) and ("group_desc" in df.columns):
                s = df["group_desc"].dropna().astype(str)
                if len(s) > 0 and s.iloc[0].strip():
                    group_title = s.iloc[0].strip()

            cols = [c for c in [
                "brand_id", "brand_name", "category", "personality_tags",
                "score", "affinity", "sim", "reason"
            ] if c in df.columns]

            df_display = df[cols].copy()

            for c in ["score", "affinity", "sim"]:
                if c in df_display.columns:
                    df_display[c] = pd.to_numeric(df_display[c], errors="coerce").round(3)

            if "category" in df_display.columns:
                def short_cat(s):
                    s = "" if pd.isna(s) else str(s)
                    for mark in ["（", "(", "[", "【"]:
                        if mark in s:
                            s = s.split(mark, 1)[0].strip()
                    if "/" in s:
                        s = s.split("/", 1)[0].strip()
                    if len(s) > 14:
                        s = s[:14] + "…"
                    return s
                df_display["category"] = df_display["category"].apply(short_cat)

        except Exception as e:
            df_display = pd.DataFrame()
            log += f"\n[stderr]\nJSON parse error: {e}"

    # 作品グループ判定HTML（要約＋補足＋注釈）
    if group_title:
        group_html = f"""
        <div style="
            border: 3px solid #111;
            border-radius: 18px;
            padding: 16px 16px;
            background: #ffffff;
            box-shadow: 0 10px 22px rgba(0,0,0,0.10);
            margin: 6px 0 12px 0;
        ">
          <div style="font-size:16px; font-weight:900; letter-spacing:0.02em;">作品グループ判定</div>
          <div style="margin-top:8px; font-size:18px; line-height:1.6;">
            今回の作品は、<span style="font-weight:900;">{html.escape(group_title)}</span>に近いと判定されました。
          </div>
          {"<div style='margin-top:8px; line-height:1.6;'>" + html.escape(group_detail) + "</div>" if group_detail else ""}
          {"<div style='margin-top:8px; font-size:13px; color:#6b7280; line-height:1.4;'>" + html.escape(cluster_note) + "</div>" if cluster_note else ""}
        </div>
        """
    else:
        group_html = """
        <div style="padding:14px; border-radius:14px; background:#fff3cd; border:1px solid #ffeeba;">
          作品グループ判定（情報を取得できませんでした。recommend.py の out_json 出力を確認してください）
        </div>
        """

    rec_cards_html = _cards_html_from_df(df_display, top_n=min(int(top_k), 10))

    return group_html, rec_cards_html, df_display, log


with gr.Blocks(title="Anime x Brand Recommender") as demo:
    gr.Markdown("# Anime x Brand Recommender UI")

    with gr.Tab("clustering"):
        gr.Markdown("## clustering")
        status = gr.Textbox(label="ファイル状況", value=list_status(), lines=10)

        with gr.Row():
            k_anime = gr.Number(label="k_anime", value=3, precision=0)
            k_brand = gr.Number(label="k_brand", value=4, precision=0)
            auto_k = gr.Checkbox(label="auto_k (silhouette探索)", value=False)
            btn_cluster = gr.Button("Run clustering")

        log_cluster = gr.Textbox(label="clustering log", lines=16)
        btn_cluster.click(ui_run_clustering, [k_anime, k_brand, auto_k], [log_cluster, status])

        gr.Markdown("---")

        weight_promo = gr.Number(label="weight_promo", value=2.0)
        btn_tieup = gr.Button("Run tieup_matrix")
        log_tieup = gr.Textbox(label="tieup_matrix log", lines=16)
        btn_tieup.click(ui_run_tieup_matrix, [weight_promo], [log_tieup, status])

    with gr.Tab("recommend"):
        gr.Markdown("## recommend")

        title = gr.Textbox(label="title", value="")
        main_genre = gr.Textbox(label="main_genre", value="")
        tone_tags = gr.Textbox(label="tone_tags", value="")
        synopsis = gr.Textbox(label="synopsis", value="", lines=4)

        with gr.Row():
            top_k = gr.Number(label="top_k", value=5, precision=0)
            top_brand_clusters = gr.Number(label="top_brand_clusters", value=3, precision=0)
            w_affinity = gr.Number(label="w_affinity", value=1.0)
            w_sim = gr.Number(label="w_sim", value=1.0)
            max_sources = gr.Number(label="max_sources", value=3, precision=0)

        btn_rec = gr.Button("Recommend", variant="primary")

        group_html = gr.HTML(
            "<div style='padding:14px; border-radius:14px; background:#f3f4f6; border:1px solid #e5e7eb;'>"
            "作品グループ判定（まだ実行していません）"
            "</div>"
        )

        rec_cards = gr.HTML(
            "<div style='padding:14px; border-radius:14px; background:#f3f4f6; border:1px solid #e5e7eb;'>"
            "推薦企業（まだ実行していません）"
            "</div>"
        )

        rec_table = gr.Dataframe(label="推薦企業（表）", wrap=True)
        rec_log = gr.Textbox(label="recommend log", lines=18)

        btn_rec.click(
            ui_run_recommend,
            [title, main_genre, tone_tags, synopsis, top_k, top_brand_clusters, w_affinity, w_sim, max_sources],
            [group_html, rec_cards, rec_table, rec_log],
        )

demo.launch(
    server_name="127.0.0.1",
    server_port=7862
)
