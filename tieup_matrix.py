# tieup_matrix.py

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight_promo", type=float, default=2.0)
    args = ap.parse_args()

    anime = pd.read_csv("anime_with_clusters.csv")
    brand = pd.read_csv("brand_with_clusters.csv")
    tieup = pd.read_csv("tieup.csv")

    print(f"Anime rows: {len(anime)}")
    print(f"Brand rows: {len(brand)}")
    print(f"Tie-up rows: {len(tieup)}")

    # 必須列チェック（余分な列は無視）
    required = {"tieup_id", "anime_id", "brand_id", "collab_group", "has_promo_video"}
    missing = required - set(tieup.columns)
    if missing:
        raise RuntimeError(f"tieup.csv missing columns: {missing}")

    # collab_group が文字列として入っている前提（product_collab等）
    tieup["has_promo_video"] = tieup["has_promo_video"].fillna(0).astype(int)
    tieup["weight"] = 1.0
    tieup.loc[tieup["has_promo_video"] == 1, "weight"] = float(args.weight_promo)

    # クラスタ付与
    tieup_merged = (
        tieup.merge(anime[["anime_id", "cluster"]].rename(columns={"cluster": "anime_cluster"}), on="anime_id", how="left")
            .merge(brand[["brand_id", "cluster"]].rename(columns={"cluster": "brand_cluster"}), on="brand_id", how="left")
    )

    print("\n--- tieup_merged (先頭5行) ---")
    show_cols = [c for c in ["tieup_id", "anime_id", "brand_id", "collab_name", "collab_type", "collab_group", "has_promo_video", "weight", "anime_cluster", "brand_cluster"] if c in tieup_merged.columns]
    print(tieup_merged[show_cols].head(5).to_string(index=False))

    # ペア集計（raw / weighted）
    grp = tieup_merged.dropna(subset=["anime_cluster", "brand_cluster"]).copy()
    grp["anime_cluster"] = grp["anime_cluster"].astype(int)
    grp["brand_cluster"] = grp["brand_cluster"].astype(int)

    raw = grp.groupby(["anime_cluster", "brand_cluster"]).size().reset_index(name="raw_count")
    weighted = grp.groupby(["anime_cluster", "brand_cluster"])["weight"].sum().reset_index(name="weighted_count")

    merged = raw.merge(weighted, on=["anime_cluster", "brand_cluster"], how="outer").fillna(0.0)

    # affinity = p(brand_cluster | anime_cluster) / p(brand_cluster)
    total_by_anime = merged.groupby("anime_cluster")["weighted_count"].sum()
    total_all = merged["weighted_count"].sum()
    total_by_brand = merged.groupby("brand_cluster")["weighted_count"].sum()

    def p_cond(row):
        denom = total_by_anime.get(row["anime_cluster"], 0.0)
        return (row["weighted_count"] / denom) if denom > 0 else 0.0

    def p_brand(row):
        denom = total_all
        numer = total_by_brand.get(row["brand_cluster"], 0.0)
        return (numer / denom) if denom > 0 else 0.0

    merged["p_cond"] = merged.apply(p_cond, axis=1)
    merged["p_brand"] = merged.apply(p_brand, axis=1)
    merged["affinity_score"] = merged.apply(lambda r: (r["p_cond"] / r["p_brand"]) if r["p_brand"] > 0 else 0.0, axis=1)

    merged.to_csv("class_pair_affinity_long.csv", index=False)

    pivot_raw = merged.pivot(index="anime_cluster", columns="brand_cluster", values="raw_count").fillna(0.0)
    pivot_weight = merged.pivot(index="anime_cluster", columns="brand_cluster", values="weighted_count").fillna(0.0)
    pivot_aff = merged.pivot(index="anime_cluster", columns="brand_cluster", values="affinity_score").fillna(0.0)

    pivot_raw.to_csv("pivot_raw_count.csv")
    pivot_weight.to_csv("pivot_weighted_count.csv")
    pivot_aff.to_csv("pivot_affinity.csv")

    print("\n保存: class_pair_affinity_long.csv, pivot_raw_count.csv, pivot_weighted_count.csv, pivot_affinity.csv")
    print(merged.sort_values(["anime_cluster", "affinity_score"], ascending=[True, False]).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
