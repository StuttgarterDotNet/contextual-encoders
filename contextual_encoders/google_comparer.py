from urllib.request import Request, urlopen
import re
import time
from .comparer import Comparer

user_agent = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS x 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.63 "
)
int_regex = r"[0-9]?[-+.,]?[0-9]+[.]?[0-9]+"
float_regex = r"\d+\.\d+"


class GoogleComparer(Comparer):
    def __init__(self):
        super().__init__(symmetric=True, multiple_values=True)

    @staticmethod
    def get_statistics(elements):
        pattern = ""
        for element in elements:
            pattern += f'"{element}"+'
        pattern = pattern[:-1]

        url = f'https://www.google.com/search?q="{pattern}"'
        request = urlopen(Request(url, headers={"User-Agent": user_agent}))
        source = request.read().decode("utf8")
        start_token = '<div id="result-stats">'
        end_token = "</div>"

        statistics = re.search(f"{start_token}(.+?){end_token}", source).group(1)
        amount = int(re.findall(int_regex, statistics)[0].replace(",", ""))
        seconds = float(re.findall(float_regex, statistics)[0])
        request.close()

        return amount, seconds

    def _compare(self, first, second):
        first_amount, first_time = GoogleComparer.get_statistics([first])
        time.sleep(2)
        second_amount, second_time = GoogleComparer.get_statistics([second])
        time.sleep(2)
        both_amount, both_time = GoogleComparer.get_statistics([first, second])

        first_value = first_amount / first_time
        second_value = second_amount / second_time
        reference_value = 0.5 * (first_value + second_value)
        both_value = both_amount / both_time

        if reference_value > both_value:
            return both_value / reference_value
        else:
            return reference_value / both_value
