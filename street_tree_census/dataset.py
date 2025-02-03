from pathlib import Path

import typer

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

import pandas as pd

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "2015-street-tree-census-tree-data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    data = pd.read_csv(input_path)
    
    data = data.dropna(subset="health").reset_index(drop=True)
    
    data["steward"] = data["steward"].fillna("0")
    data["guards"] = data["guards"].fillna("no guards")
    data["root_problems"] = (
        data[["root_stone", "root_grate", "root_other"]].eq("Yes").sum(axis=1)
    )
    data["trunk_problems"] = (
        data[["trunk_wire", "trnk_light", "trnk_other"]].eq("Yes").sum(axis=1)
    )
    data["brch_problems"] = (
        data[["brch_light", "brch_shoe", "brch_other"]].eq("Yes").sum(axis=1)
    )
    data["problems"] = data[["root_problems", "trunk_problems", "brch_problems"]].sum(axis=1)
    data["ct"] = data["boro_ct"] % 1000000 

    columns_to_drop = [
        "tree_id",  # unique
        "stump_diam",  # constant
        "status",  # constant
        "state",  # constant
        "spc_latin",  # same as "spc_common"
        "borocode",  # same as "borough"
        "nta_name",  # same as "nta"
        "longitude",  # same as "x_sp"
        "latitude",  # same as "y_sp"
        "council district",  # same as "cncldist"
        "census tract",  # same as boro_ct
        "zip_city",  # almost same as "borough", corr with "postcode"
        "st_assem",  # corr with "st_senate"
        "bbl",  # corr with "boro_ct"
        "bin",  # corr with "boro_ct"
        "community board",  # corr with "boro_ct"
        "address",  # много уникальных значений
        "created_at",  # нам это не важно
        "root_stone",
        "root_grate",
        "root_other",
        "trunk_wire",
        "trnk_light",
        "trnk_other",
        "brch_light",
        "brch_shoe",
        "brch_other",
        "boro_ct",
        # "borough",   # same as "borough"
    ]   
    data = data.drop(columns=columns_to_drop)

    mapping = {"0": 0, "1or2": 1.5, "3or4": 3.5}
    data["steward"] = data["steward"].map(lambda x: mapping.get(x, 5))
    
    data = data.loc[data["postcode"] != 83]
    data = data.loc[data["tree_dbh"] <= 50]
    data = data.drop_duplicates()
    
    NTA_BOROUGH_SIZE = 2
    data["nta"] = data.nta.str[NTA_BOROUGH_SIZE:].astype(int)

    most_frequent_spc = data["spc_common"].value_counts().idxmax()
    data["spc_common"] = data["spc_common"].fillna(most_frequent_spc)

    most_frequent_sidewalk = data["sidewalk"].value_counts().idxmax()
    data["sidewalk"] = data["sidewalk"].fillna(most_frequent_sidewalk)

    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
