from abc import abstractclassmethod
from typing import Optional
import requests

import openai
from langchain import PromptTemplate
from googletrans import Translator
# from TTS.api import TTS
import datetime as DT
import pytz
import random
import math
import json
import datetime

from utils import get_args

import sys
sys.set_int_max_str_digits(0)

API_KEY = "OPENAI_API_KEY"
ORG = "OPENAI_ORG"

class BaseAPI:
    def __init__(
        self,
        name: str, # the name of the API call
        description: str, # the natural language description of the API call
        prompt_template: PromptTemplate, # API usage guide
        pattern: str=None # output pattern
    ):
        self.name = name
        self.description = description
        self.prompt_template = prompt_template
        self.pattern = pattern
        self.mocked_result = None

    @abstractclassmethod
    def execute(self):
        pass
    
    def __call__(self, *args: str, **kargs: str) -> Optional[str]:
        if "mock" in kargs and kargs["mock"] and self.mocked_result is not None:
            return self.mocked_result

        kargs.pop("mock", None)

        try:
            output = self.execute(*args, **kargs)
        except Exception as e:
            print(f"API {self.name} failed with error: {e}, args: {args}, kwargs: {kargs}")
            return None

        return str(output) if output is not None else None


def CreateAPIPipeline(api_class_dict: dict, class_name: str="APIPipeline", exec_args: dict = {}):
    '''
    Returns a new API class that can execute multiple APIs in a pipeline.
    api_class_dict: a dictionary of API classes to be executed in the pipeline, with the key being the identifier of the API.
    class_name: the name of the new API class.
    exec_args: a dictionary of API identifiers to a dictionary of arguments to be passed to the API's execute function. The key of the inner dictionary is the argument name of the API's execute function, and the value is the argument name of the APIPipeline's execute function.

    Example:\n
    MultiLingualAPI = CreateAPIPipeline({"MT": MTAPI, "ContextQA": ContextQAAPI}, "MultiLingualAPI", exec_args={"MT": {"dest": "mtdest", "src": "mtsrc"}})
    api = MultiLingualAPI("Multi-lingual QA",  multilingual_description)
    api(input_query,
        mtdest = 'en',
        mtsrc = 'zh-CN') 
    '''
    new_class = type(class_name, (BaseAPI,), {})
    def init(self, name: str, description: str, prompt_template: PromptTemplate=None, pattern: str = None):
        self.name = name
        self.description = description
        self.prompt_template = prompt_template
        self.pattern = pattern

        self.apis = {}
        self.exec_args = exec_args
        self.apis = api_class_dict
        for k, v in api_class_dict.items():
            self.apis[k] = v(name, description, prompt_template, pattern)

    def execute(self, input_query: str, **kwargs):
        try:
            ret = input_query
            for k, v in self.apis.items():
                if k in self.exec_args:
                    curr_args = {}
                    for arg, mapped_arg in self.exec_args[k].items():
                        curr_args[arg] = kwargs[mapped_arg]
                    ret = v(ret, **curr_args)
                else:
                    ret = v(ret)
            return ret

        except Exception as e:
            return None
       
    setattr(new_class, "__init__", init)
    setattr(new_class, "execute", execute)
    return new_class

class CalculatorAPI(BaseAPI):
    def execute(self, input: str):
        try:
            return round(eval(input), 3)
        except:
            return None

class WikiSearchAPI(BaseAPI):
    def colbertv2_get_request(self, url: str, query: str, k: int=1):
        payload = {"query": query, "k": k}
        res = requests.get(url, params=payload)

        topk = res.json()["topk"][:k]
        return topk

    def execute(self, input_query: str):
        try:
            url = "http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search"
            MAX_TOKENS = 300 # max length of string to return
            topk = self.colbertv2_get_request(url, input_query)
            output = [doc["text"] for doc in topk][0]
            start_idx = output.find('|') + 2
            return output[start_idx:start_idx + MAX_TOKENS]
        except:
            return None

class QAAPI(BaseAPI):
    def __init__(self, name: str, description: str, prompt_template: PromptTemplate, pattern: str = None):
        super().__init__(name, description, prompt_template, pattern)
        self.prompt = [
            {"role": "system", "content": "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\"."},
            {"role": "user", "content": "You are the Question Answering tool that answers questions by reasoning and commonsense knowledge. Here are some examples of questions you can answer:"},
            {"role": "user", "content": " Where do adults use glue sticks? A: classroom B: desk drawer C: at school D: office E: kitchen drawer"},
            {"role": "assistant", "content": "Glue sticks are commonly used by adults in office settings for various tasks, so the answer is D: office."},
            {"role": "user", "content": "What could go on top of wood? A: lumberyard B: synagogue C: floor D: carpet E: hardware store"},
            {"role": "assistant", "content": "Wood is commonly used as a material for flooring,  therefore only the option D: carpet among all these options can go on top of wood floors."},
            {"role": "user", "content": "The women met for coffee. What was the cause of this? A: The cafe reopened in a new location. B: They wanted to catch up with each other."},
            {"role": "assistant", "content": "Considering the options, the more likely cause for the women meeting for coffee would be B: They wanted to catch up with each other. Meeting for coffee is often chosen as a way to have a relaxed and informal conversation, providing an opportunity for friends or acquaintances to reconnect and share updates about their lives."},
            {"role": "user", "content": "What is the square root of banana?"},
            {"role": "assistant", "content": "Unknown"}
        ]

    def get_input_query(self, input_query: str):
        ret = self.prompt.copy()
        ret.append({"role": "user", "content": input_query})
        return ret

    def execute(self, input_query: str):
        try: 
            openai.api_key = API_KEY
            ret = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                max_tokens=300,
                temperature=0.2,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.1,
                messages=self.get_input_query(input_query)
            )
        except Exception as e:
                # return e.__str__()
                return None
        return ret.choices[0].message.content

        
class ContextQAAPI(BaseAPI):
    def __init__(self, name: str, description: str, prompt_template: PromptTemplate, pattern: str = None):
        super().__init__(name, description, prompt_template, pattern)
        self.prompt = [
            {"role": "system", "content": "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\"."},
            {"role": "user", "content": "You are the Context Question Answering tool that answers questions according to the given context."},
        ]

    def get_input_query(self, input_query: str):
        ret = self.prompt.copy()
        ret.append({"role": "user", "content": input_query})
        return ret

    def execute(self, input_query: str):
        try: 
            openai.api_key = API_KEY
            ret = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                max_tokens=300,
                temperature=0.2,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.1,
                messages=self.get_input_query(input_query)
            )
        except Exception as e:
                return None
        return ret.choices[0].message.content

class TimeZoneAPI(BaseAPI):
    def execute(self, input_query: str, src: str, dest: str):
        try:
            src = pytz.timezone(src)
            dest = pytz.timezone(dest)
            return src.localize(DT.datetime.strptime(input_query, "%Y-%m-%d %H:%M:%S")).astimezone(dest).strftime("%Y-%m-%d %H:%M:%S")
        except:
            return None

class MTAPI(BaseAPI):
    def __init__(self, name: str, description: str, prompt_template: PromptTemplate, pattern: str = None):
        super().__init__(name, description, prompt_template, pattern)
        self.translator = Translator(service_urls=[
                'translate.google.com'
                ])
    def execute(self, text: str, dest: str, src: str = None):
        try:
            src = self.translator.detect(text).lang if src is None else src
            a = self.translator.translate(text, src=src, dest=dest)
        except Exception as e:
            return None
        if a is not None:
            return a.text
        else:
            print("MT Error: None translation")
            return None
              
# multi-lingual QA model
MultiLingualAPI = CreateAPIPipeline({"MT": MTAPI, "ContextQA": ContextQAAPI}, "MultiLingualAPI", exec_args={"MT": {"dest": "mtdest", "src": "mtsrc"}})

class Sleep(BaseAPI):
    def execute(self, input: str):
        # time.sleep(int(input))
        try:
            input = float(input)
            if input < 0:
                return None
            else:
                return "sleep for " + str(input) + " seconds." # mock response
        except:
            return None
    
class RobotMove(BaseAPI):
    def execute(self, input: str):
        try:
            input = float(input)
            if input < 0:
                return "Robot moving failed."
            if random.random() < 0.9:
                return "Robot is moving forward for " + str(input) + " meters." # mock response
            else:
                return "Moving failed. Robot is stuck."
        except:
            return None
        
class Pow(BaseAPI):
    def execute(self, a: str, b: str):
        try:
            if float(a) > 20 and int(b) > 10:
                return None
            else:
                return float(a) ** int(b)
        except:
            return None
        
class Log(BaseAPI):
    def execute(self, a: str, b: str):
        try:
            return math.log(int(a), int(b))
        except:
            return None

class ImageGenerationAPI(BaseAPI):
    def __init__(self, name: str, description: str, prompt_template: PromptTemplate, pattern: str = None):
        super().__init__(name, description, prompt_template)
        self.mocked_result = "https://www.mocked-url.com/2.jpg"
        self.prev_org = None
        self.prev_key = None
        
    def execute(self, input_query: str, **kwargs):
        self.prev_org = openai.organization
        self.prev_key = openai.api_key

        openai.api_key = API_KEY
        openai.organization = ORG

        try:
            response = openai.Image.create(
                prompt=input_query,
                n=1,
                size="256x256"
            )
            img_url = response["data"][0]["url"]
            openai.api_key = self.prev_key
            openai.organization = self.prev_org
            return img_url
        except Exception as e:
            openai.api_key = self.prev_key
            openai.organization = self.prev_org
            raise e

class LocationSearchAPI(BaseAPI):
    def __init__(self, name: str, description: str, prompt_template: PromptTemplate, pattern: str = None):
        super().__init__(name, description, prompt_template)

        self.subscription = "SUBSCRIPTION_KEY"

    def _get_detail(self, name, lat=None, lon=None, countrySet="CN,JP,US,CA,HK,JP"):
        url = f"https://atlas.microsoft.com/search/fuzzy/json"
        params = {
            "subscription-key": self.subscription,
            "api-version": 1,
            "query": name,
            "countrySet": countrySet,
            "limit": 1,
        }

        if lat is not None and lon is not None:
            params['lat'] = lat
            params['lon'] = lon
            params['radius'] = 100000

        response = requests.get(url, params=params)
        data = json.loads(response.text)
        return data
    
    def _get_city(self, city, countrySet="US,CN,CA,HK,JP"):
        url = f"https://atlas.microsoft.com/search/address/json"
        params = {
            "subscription-key": self.subscription,
            "api-version": 1,
            "language": "en_US",
            "query": city,
            "countrySet": countrySet,
            "limit": 1,
        }
        response = requests.get(url, params=params)
        data = json.loads(response.text)
        return data
    
    def execute(self, input_query: str, city: Optional[str] = None, countrySet="CN,JP,US,CA,HK,JP"):
        try:
            lat = None
            lon = None
            if city is not None:
                data = self._get_city(city, countrySet)
                lat = data['results'][0]['position']['lat']
                lon = data['results'][0]['position']['lon']

            data = self._get_detail(input_query, lat, lon, countrySet)
            if len(data['results']) == 0:
                return "Did not find any result."
            result = data['results'][0]
            name = result['poi']['name']
            address = result['address']['freeformAddress']
            return f"{name}: {address}"
        except Exception as e:
            raise e
        
class WeatherAPI(BaseAPI):
    def __init__(self, name: str, description: str, prompt_template: PromptTemplate, pattern: str = None):
        super().__init__(name, description, prompt_template)
        self.subscription = "SUBSCRIPTION_KEY"

    def _get_city(self, city, countrySet="US,CN,CA,HK,JP"):
        url = f"https://atlas.microsoft.com/search/address/json"
        params = {
            "subscription-key": self.subscription,
            "api-version": 1,
            "language": "en_US",
            "query": city,
            "countrySet": countrySet,
            "limit": 1
        }
        response = requests.get(url, params=params)
        data = json.loads(response.text)
        return data
    
    def _get_weather(self, lat, long):
        url = f"https://atlas.microsoft.com/weather/currentConditions/json"
        params = {
            "subscription-key": self.subscription,
            "api-version": 1,
            "query": f"{lat},{long}",
            "limit": 1
        }
        response = requests.get(url, params=params)
        data = json.loads(response.text)
        return data
    
    def execute(self, input_query: str, countrySet="US,CN,CA,HK,JP"):
        try:
            data = self._get_city(input_query, countrySet)
            lat = data['results'][0]['position']['lat']
            long = data['results'][0]['position']['lon']
            # print(lat, long)
            weather = self._get_weather(lat, long)['results'][0]
            at_time = weather['dateTime']
            tempe = f"{weather['temperature']['value']}\u00b0{weather['temperature']['unit']}"
            felltemp = f"{weather['realFeelTemperature']['value']}\u00b0{weather['realFeelTemperature']['unit']}"
            humid = f"{weather['relativeHumidity']}%"
            visi = f"{weather['visibility']['value']}{weather['visibility']['unit']}"
            phrase = weather['phrase']
            return f"Time: {at_time}, Temperature: {tempe}, Feels like: {felltemp}, Humidity: {humid}, Visibility: {visi}, Weather: {phrase}"
        except Exception as e:
            raise e
        
class TimeAPI(BaseAPI):
    def execute(self):
        return datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S, %Z")