import json

# Load flower info
with open(r"C:\ai_project\codes\flower_info.json", "r") as f:
    flower_data = json.load(f)

def suggest_property_similar(predicted_label, top_n=5):
    """
    Suggest flowers with similar care needs/usage.
    """
    if predicted_label not in flower_data:
        return []

    target_props = flower_data[predicted_label].get("properties", {})
    if not target_props:
        return []

    scores = []
    for flower, info in flower_data.items():
        if flower == predicted_label:
            continue
        props = info.get("properties", {})
        score = 0

        # Compare common properties
        for key in ["sunlight", "watering", "soil", "climate"]:
            if key in target_props and key in props and target_props[key] == props[key]:
                score += 1
        scores.append((flower, score))

    # Sort by highest similarity
    scores.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in scores[:top_n]]
