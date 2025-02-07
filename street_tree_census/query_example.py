import requests

if __name__ == "__main__":
    features = {
        "block_id": 348711,
        "steward": 0.0,
        "tree_dbh": 3,
        "postcode": 11375,
        "cncldist": 29,
        "st_senate": 16,
        "nta": 17,
        "ct": 0,
        "x_sp": 1027431.148,
        "y_sp": 202756.7687,
        "problems": 0,
        "root_problems": 0,
        "trunk_problems": 0,
        "brch_problems": 0,
        "curb_loc": "OnCurb",
        "spc_common": "red maple",
        "guards": "no guards",
        "sidewalk": "NoDamage",
        "user_type": "TreesCount Staff",
        "borough": "Queens",
    }
    resp = requests.post("http://127.0.0.1:80/predict", json=features)
    print(resp.json())
