from langchain import PromptTemplate

zero_shot_prompt = """You are only asked to return answers, do not show the intermediate steps.
Input: {input}
Output:"""
zero_shot_prompt = PromptTemplate(input_variables=["input"], template = zero_shot_prompt)

calculator_description = "Calculator API is used for answering questions that contain numbers and require arithemtic operations, including addition, subtraction, multiplication, division."
calculator_prompt = """Calculator API is used for solving questions that require arithemtic operations, including addition, subtraction, multiplication, division. You task is to rephrase the question prepended by the special token <Q> and generate Calculator API call prepended by <API> for solving that question.
You can call the API by writing "[Calculator(formula)]" where "formula" is the arithmetical formula you want to solve. Here are some examples of Calculator API calls:

Input: Rangers from Flora Natural Park and Wildlife Reserve also joined the activities on that day. They planted 75 redwood trees and 25 cypress trees to replace the trees that were destroyed during a recent forest fire. How many trees did the rangers plant?
Output: <Q> Rangers planted 75 redwood trees and 25 cypress trees. So there are 75 + 25 trees Rangers planted. <API> [Calculator(75 + 25)].

Input: There were 86 pineapples in a store. The owner sold 48 pineapples. 9 of the remaining pineapples were rotten and thrown away. How many fresh pineapples are left?
Output: <Q> There are total 86 pineapples. 48 pineapples are sold out, so there are 86 - 48 pineapples now. 9 of the remaining are thrown away, so there are 86 - 48 - 9 pineapples. <API> [Calculator(86 - 48 - 9)].

Input: Sarah is making bead necklaces. She has 945 beads and is making 7 necklaces with each necklace using the same number of beads. How many beads will each necklace use?
Output: <Q> Sarah has 945 beads and is going to make 7 necklaces with each necklacec using the same number of breads, so each necklace will use 945 / 7 beads.  <API> [Calculator(945 / 7)].

Input: A movie poster was 4 inches wide and 7 inches tall. What is the area of the poster?
Output: <Q> The area is computed by the production of the width and the height. The width is 4 inches and the height is 7 inches. So the area is 4 * 7. <API> [Calculator(4 * 7)].

Input: There were sixty-one people in line at lunch when twenty-two more got in line. How many people were there total in line?
Output: <Q> There are sixty-one people, 61, people in line and twenty-two, 22, more got in line. Therefore, there are 61 + 22 people total in line. <API> [Calculator(61 + 22)].

Input: {input}
Output:"""

calculator_prompt = PromptTemplate(input_variables=["input"], template = calculator_prompt)

qa_description = "Question Answering API helps you get additional information and commonsense required to answer questions."
qa_prompt = """Question Answering API helps you get additional information and commonsense required to answer questions. You task is to rephrase the question prepended by the special token <Q> and generate QA API call prepended by <API> for solving that question. Here are some examples of API calls:
You can call the API by writing "[QA(question)]" where "question" is the question you want to ask. Here are some examples of QA API calls:

Input: The man broke his toe. What was the cause of this? A: He got a hole in his sock. B: He dropped a hammer on his foot.
Output: <Q> The man broke his toe. What was the cause of this? A: He got a hole in his sock. B: He dropped a hammer on his foot. <API> [QA("The man broke his toe. What was the CAUSE of this? A: He got a hole in his sock B: He dropped a hammer on his foot")].

Input: What do people want to acquire from opening business? A: home B: wealth C: bankruptcy D: get rich
Output: <Q>  What do people want to acquire from opening business? A: home B: wealth C: bankruptcy D: get rich <API> [QA("What do people want to acquire from opening business? A: home B: wealth C: bankruptcy D: get rich")].

Input: What other name is Coca-Cola known by? A: Coke B: Cola C: Pepsi D: Diet Pepsi
Output: <Q> What other name is Coca-Cola known by? Choose the most reasonable option from A: Coke B: Cola C: Pepsi D: Diet Pepsi <API> [QA("What other name is Coca-Cola known by? A: Coke B: Cola C: Pepsi D: Diet Pepsi")].

Input: {input}
Output:"""
qa_prompt = PromptTemplate(input_variables=["input"], template = qa_prompt)

wiki_description = "Wikipedia Search API is to look up information from Wikipedia that is necessary to answer the question."
wiki_prompt = """Wikipedia Search API is to look up information from Wikipedia that is necessary to answer the question. You task is to rephrase the question prepended by the special token <Q> and generate Wikipedia search API call prepended by <API> for solving that question.
You can do so by writing "[WikiSearch(term)]" where "term" is the search term you want to look up. Here are some examples of WikiSearch API calls:

Input: The colors on the flag of Ghana have the following meanings: green for forests, and gold for mineral wealth. What is the meaning of red?
Output: <Q> Ghana flag green means forests, Ghana flag gold means mineral wealth, what is the the meaning of Ghana flag red? <API> [WikiSearch("Ghana flag red meaning")].

Input: What are the risks during production of nanomaterials?
Output: <Q> What is the risk during production of nanomaterials? <API> [WikiSearch("nanomaterial production risks")].

Input: Metformin is the first-line drug for which disease?.
Output: <Q> Metformin is the first-line drug used to treat which disease? <API> [WikiSearch("Metformin first-line drug")].

Input: {input}
Output:"""

wiki_prompt = PromptTemplate(input_variables=["input"], template = wiki_prompt)

mt_description = "Machine Translation API is used for translating text from one language to another."

mt_prompt = """Machine Translation API is used for translating text from one language to another. You task is to rephrase the question prepended by the special token <Q> and generate MT API call prepended by <API> for solving that question.
You can do so by writing "[MT(text, target_language)]" where "text" is the text to be translated and "target_language" is the language to translate to. Here are some examples of MT API calls:

Input: What is 自然语言处理 in English.
Output: <Q> Translate "自然语言处理" to English. <API> [MT("自然语言处理", "en")].

Input: How do I ask Japanese students if they had their dinner yet?
Output: <Q> Translate "Did you have dinner yet" in Japanese <API> [MT("Did you have dinner yet?", "ja")].

Input: How to express I love you in Franch?
Output: <Q> Translate "I love you" in Franch? <API> [MT("I love you", "fr")].

Input: {input}
Output:"""

mt_prompt = PromptTemplate(input_variables=["input"], template = mt_prompt)

tts_description = "Text to Speech API is used for converting text to speech."
tts_prompt = """Text to Speech API is used for converting text to speech. You task is to rephrase the question prepended by the special token <Q> and generate TTS API call prepended by <API> for solving that question.
You can do so by writing "[TTS(text)]" where "text" is the text to be converted. Here are some examples of TTS API calls:

Input: Please read the following text: "The quick brown fox jumps over the lazy dog."
Output: <Q> Text to Speech for: "The quick brown fox jumps over the lazy dog." <API> [TTS("The quick brown fox jumps over the lazy dog.")].

Input: How to pronounce: "Pneumonoultramicroscopicsilicovolcanoconiosis"?
Output: <Q> Text to Speech for: "Pneumonoultramicroscopicsilicovolcanoconiosis" <API> [TTS("Pneumonoultramicroscopicsilicovolcanoconiosis")].

Input: Please say I love you.
Output: <Q> Text to Speech for: "I love you" <API> [TTS("I love you")].

Input: {input}
Output:"""

tts_prompt = PromptTemplate(input_variables=["input"], template = tts_prompt)

timezoneconverter_description = "Timezone Converter API is used for converting time between different timezones."
timezoneconverter_prompt = """Timezone Converter API is used for converting time between different timezones. You task is to rephrase the question prepended by the special token <Q> and generate Timezone Converter API call prepended by <API> for solving that question.
You can do so by writing "[TimezoneConverter(time, from_timezone, to_timezone)]" where "time" is the time to be converted, "from_timezone" is the timezone of the input time, and "to_timezone" is the timezone to convert to. Here are some examples of Timezone Converter API calls:

Input: Convert 2000-04-01 16:08:23 from EST to EDT.
Output: <Q> Time is 2000-04-01 16:08:23, the source is EST, the target is EDT. <API> [TimezoneConverter("2000-04-01 16:08:23", "EST", "EDT")].

Input: It is 22:00, January 2nd, 2022 in Shanghai, what time is it in New York?
Output: <Q> Time is 2022-01-02 22:00:00, the source is Beijing, so the timezone is Asia/Shanghai, the target is New York, so the timezone is America/New_York. <API> [TimezoneConverter("2022-01-02 22:00:00", "Asia/Shanghai", "America/New_York")].

Input: {input}
Output:"""
timezoneconverter_prompt = PromptTemplate(input_variables=["input"], template = timezoneconverter_prompt)

# conetext QA only used for multilingual QA
context_qa_description = "Contextual Question Answering API retrieves answers from the given context."
multilingual_description = "Multilingual QA API is used for questions where the context is in English, while the question is in other language."

sleep_description = "Sleep API is used for pausing the program for a given time"
sleep_prompt = """Sleep API is used for pausing the program for a given time. You task is to rephrase the request prepended by the special token <Q> and generate Sleep API call prepended by <API>.
You can do so by writing "[Sleep(time)]" where "time" is the time Program has to be paused. Here are some exampels of Sleep API calls:

Input: Sleep for 20 seconds.
Output: <Q> Program sleeps for 20 seconds. <API> [Sleep(20)].

Input: Stop executing program for 1 minutes.
Output: <Q> Program sleeps for 1 minutes which is 60 seconds. <API> [Sleep(60)].

Input: {input}
Output:"""
sleep_prompt = PromptTemplate(input_variables=["input"], template = sleep_prompt)

robot_description = "Robot Move API is used for controlling the robot to move forward"
robot_prompt = """Robot Move API is used for controlling the robot to move forward for n seconds. You task is to rephrase the request prepended by the special token <Q> and generate Robot Move API call prepended by <API>.
You can do so by writing "[RobotMove(time)]" where "time" is the time Robot has to move. Here are some exampels of Robot Move API calls:

Input: I want the robot to move forward for 21.3 meters.
Output: <Q> Robot moves forward for 21.3 meters. <API> [RobotMove(21.3)].

Input: Move the robot forward for 30 centimeters.
Output: <Q> Robot moves forward for 30 centimeters which is 0.3 meters . <API> [RobotMove(0.3)].

Input: {input}
Output:"""
robot_prompt = PromptTemplate(input_variables=["input"], template = robot_prompt)

pow_description = "Pow API is used for calculating the power of a number"
pow_prompt = """Pow API is used for calculating the power of a number. You task is to rephrase the request prepended by the special token <Q> and generate Pow API call prepended by <API>.
You can do so by writing "[Pow(base, power)]" where "base" is the base number and "power" is the power. Here are some exampels of Pow API calls:

Input: What is 2 to the power of 3?
Output: <Q> 2^3, 2 is the base and 3 is the power. <API> [Pow(2, 3)].

Input: I want the 4 to be cubed. What is the result?
Output: <Q> 4^3, 4 is the base and 3 is the power. <API> [Pow(4, 3)].

Input: {input}
Output:"""
pow_prompt = PromptTemplate(input_variables=["input"], template = pow_prompt)

log_description = "Log API is used for calculating the log of a number"
log_prompt = """Log API is used for calculating the log of a number. You task is to rephrase the request prepended by the special token <Q> and generate Log API call prepended by <API>.
You can do so by writing "[Log(base, number)]" where "base" is the base number and "number" is the number. Here are some exampels of Log API calls:

Input: What is log base 2 of 8?
Output: <Q> log base 2 of 8, 2 is the base and 8 is the number. <API> [Log(2, 8)].

Input: I want the log base 4 of 16. What is the result?
Output: <Q> log base 4 of 16, 4 is the base and 16 is the number. <API> [Log(4, 16)].

Input: {input}
Output:"""

image_description = "Image Generation API generates images from text descriptions."
image_prompt = """Image Generation API generates images from text descriptions. You task is to rephrase the question prepended by the special token <Q> and generate Image Generation API call prepended by <API> for solving that question. 
You can do so by writing "[ImageGeneration(text)]" where "text" is the text to be translated. Here are some examples of Image Generation API calls:

Input: a white siamese cat
Output: <Q> Generate an image of a white siamese cat. <API> [ImageGeneration("a white siamese cat")].

Input: I want to see an image about beautiful sunset.
Output: <Q> Generate an image of beautiful sunset. <API> [ImageGeneration("beautiful sunset")].

Input: Generate an image of a tall tree.
Output: <Q> Generate an image of a tall tree. <API> [ImageGeneration("a tall tree")].

Input: {input}
Output:"""

image_prompt = PromptTemplate(input_variables=["input"], template = image_prompt)

location_search_description = "Location Search API is used for getting location information of a place."
location_search_prompt = """Location Search API is used for getting location information of a place. You task is to rephrase the question prepended by the special token <Q> and generate Location Search API call prepended by <API> for solving that question.
You can do so by writing "[LocationSearch(place, city, country)]" where "place" is the place, "city" is the city and "country" is the country. Here are some examples of Location Search API calls:

Input: How can I get to the University of Hong Kong? 
Output: <Q> How can I get to the University of Hong Kong? <API> [LocationSearch("University of Hong Kong", "Hong Kong", "HK")].

Input: Give me a sushi restaurant in Tokyo.
Output: <Q> Give me a sushi restaurant in Tokyo. <API> [LocationSearch("Sushi Restaurant", "Tokyo", "JP")].

Input: Find me a hotel in Baltimore.
Output: <Q> Find me a hotel in Baltimore. <API> [LocationSearch("Hotel", "Baltimore", "US")].

Input: {input}
Output:"""

location_search_prompt = PromptTemplate(input_variables=["input"], template = location_search_prompt)

weather_description = "Weather API is used for getting weather information of a location."
weather_prompt = """Weather API is used for getting weather information of a specific location. You task is to rephrase the question prepended by the special token <Q> and generate Weather API call prepended by <API> for solving that question.
You can do so by writing "[Weather(location, country)]" where "location" is the location and "country" is the country. Here are some examples of Weather API calls:

Input: What is the weather in Baltimore?
Output: <Q> Baltimore is located in Maryland, US. What is the weather in Baltimore? <API> [Weather("Baltimore, MD", "US")].

Input: I want to go to the beach, is it sunny in Miami?
Output: <Q> Miami is located in Florida, US. Is it sunny in Miami? <API> [Weather("Miami, FL", "US")].

Input: I want to know the weather in Hong Kong.
Output: <Q> Hong Kong is located in Hong Kong, CN. What is the weather in Hong Kong? <API> [Weather("Hong Kong, HK", "HK")].

Input: {input}
Output:"""

weather_prompt = PromptTemplate(input_variables=["input"], template = weather_prompt)

time_description = "Time API is used for getting the current time with timezone."
time_prompt = """Time API is used for getting the current time with timezone. You task is to rephrase the question prepended by the special token <Q> and generate Time API call prepended by <API> for solving that question. You can do so by writing "[Time()]"
Here are some examples of Time API calls:

Input: What is the current time?
Output: <Q> What is the current time? <API> [Time()]

Input: What is the time in Hong Kong?
Output: <Q> What is the time in Hong Kong? <API> [Time()]

Input: {input}
Output:"""

time_prompt = PromptTemplate(input_variables=["input"], template = time_prompt)