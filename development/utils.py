import csv

def load_teachers(filepath="./development/teachers.csv"):
    teachers = []
    with open(filepath, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            teachers.append({
                "id": row["id"],
                "name": row["name"],
                "office": row["office"] if row["office"] else False,
                "position": row["position"],
                "background": row["background"],
                "info": row["info"]
            })
    return teachers


def prepare_teacher_vectors(teachers, embed_model):
    vectors = []
    for teacher in teachers:
        vector = embed_model.encode(teacher["info"]).tolist()
        vectors.append({
            "id": teacher["id"],
            "values": vector,
            "metadata": {
                "name": teacher["name"],
                "office": teacher["office"],
                "position": teacher["position"],
                "background": teacher["background"],
                "info": teacher["info"]
            }
        })
    return vectors
