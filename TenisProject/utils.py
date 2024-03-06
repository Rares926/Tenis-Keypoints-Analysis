import json 
import copy 
from typing import List, Dict

def load_chunks(json_path: str,
                chunk_type: str):

    with open(json_path, 'r') as file:
        json_content = file.read()

    json_data = json.loads(json_content)

    chunks = []
    for serve in json_data["classes"][chunk_type]:
        chunks.append([serve["start"],serve["end"],serve["name"]])

    return chunks

def map_serve_to_hit(serves: List[Dict], hits: List[Dict], ace_threshold: int = 400):
    """
    Maps each serve to a hit based on certain rules.
    INFO:
        If serve is In we either save the serve-hit tuple as (serve_name, hit_name) or 
            as (serve_name, 'Ace or Bad retrieval) based on the ace threshold.
        If serve is not In the tuple is saved as (serve_name, serve_result)

    Args:
        serves (List[Dict]): List of dictionaries, each representing a serve with details
            including frame timing in the match and outcome of the Serve.
        hits (List[Dict]): List containing dictionaries for each Hit with details such as
            type of hit(Forehand, Backhand, etc) side of Hit(Flat, Topspin, Slice etc) and
            frame timing of the Hit.
        ace_threshold (int, optional): Maximum time difference beetwen a serve end and a
            hit in order to consider it as a succesfull serve that was not returned(Ace)
            or a bad retrieval. Defaults to 400.

    Returns:
        List[Tuple]: List of all served mapped to their following hit.
    """
    serve_to_hit_mapping = {}

    hits_copy = copy.deepcopy(hits)
    for serve in serves:
        closest_hit = None
        closest_time_diff = float('inf')

        if serve["custom"]["Result"] == "In":
            # Check for hits following the serve
            for hit in hits_copy:
                time_diff = abs(serve['end'] - hit['start'])
                
                # save closest hit start to our serve end
                if time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    closest_hit = hit

            # Determine if serve is an ace
            if closest_hit and closest_time_diff <= ace_threshold:
                serve_to_hit_mapping[serve['name']] = closest_hit['name']
                hits_copy.remove(closest_hit)  # Eliminating the matched hit
            else:
                serve_to_hit_mapping[serve['name']] = "Ace or Bad retrieval"
        else:
            serve_to_hit_mapping[serve['name']] = serve["custom"]["Result"]

    return serve_to_hit_mapping

def load_combined_chunks(json_path: str,
                         chunk_type: tuple):
    """
    Combines two frames from two types of chunks.

    Args:
        json_path (str): Path to the json file that retaions info about a tennis match.
        chunk_type (str): A string specifying the types of chunks to be combined,
            indicating the keys in the json_data["classes"] dictionary.

    Returns:
        List[(int,int,str)]: List with frame start frame end and chunk name for each chunk combination.
    """
    with open(json_path, 'r') as file:
        json_content = file.read()

    json_data = json.loads(json_content)
    chunks_mapping = map_serve_to_hit(json_data["classes"][chunk_type[0]],
                                      json_data["classes"][chunk_type[1]])

    chunk_2_valid_ids = [item for item in chunks_mapping.values() if item.isdigit()]
    chunks = []
    for serve in json_data["classes"][chunk_type[0]]:
        end_chunk_id = chunks_mapping[serve["name"]]
        if end_chunk_id not in chunk_2_valid_ids:
            chunks.append([serve["start"],serve["end"],serve["name"]])
        else:
            chunk2 = [item for item in json_data["classes"][chunk_type[1]] if item['name'] == end_chunk_id][0]
            chunks.append([serve["start"],chunk2["end"],serve["name"]])
    return chunks


# serve_chunks = load_combined_chunks("data/V010.json", chunk_type=["Serve","Hit"])
# print(serve_chunks)