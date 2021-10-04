import random
import string
from typing import List


def recommend(query: List[str], limit: int = 5) -> List[str]:
    if len(query) < limit:
        # generate random strings
        for _ in range(limit - len(query)):
            query.append("".join(random.sample(string.ascii_letters, 7)))
    output = query[:limit]
    return output


if __name__ == "__main__":
    print(recommend([]))
