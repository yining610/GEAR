import openai
openai.api_key = "OPENAI_API_KEY"

class Message():
    def __init__(self, content, role="user", name = None, direct_to_me=True) -> None:
        self.role = role
        self.content = content
        self.name = name
        self.direct_to_me = direct_to_me

    def get_dict(self) -> dict:
        ret = {
            "role": self.role,
            "content": self.content
        }
        if self.name:
            ret["name"] = self.name
        return ret
    

class OpenAIChatGPT():

    def __init__(self, prompt_template = None) -> None:
        self.max_price = 0.3
        self.indirect_message_price = 0.05
        self.init_message = []

        if (prompt_template is None):
            self.reset_session()
            return
        
        for line in prompt_template.template.splitlines():
            if len(line.strip()) == 0:
                continue
            if line.startswith("Input: {input}"):
                break
            if line.startswith("Input:"):
                self.init_message.append(self.new_message("system", line.replace("Input:", "").strip(), "example_user"))
            elif line.startswith("Output:"):
                self.init_message.append(self.new_message("system", line.replace("Output:", "").strip(), "example_assistant"))
            else:
                self.init_message.append(self.new_message("system", line.strip()))
        self.reset_session()

    def new_message(self, role, content, name = None, direct_to_me=True) -> dict:
        return Message(content, role, name, direct_to_me)

    def reset_session(self) -> None:
        self.messages = self.init_message.copy()
        self.indirect_messages = []

    def get_messages_list(self, messages) -> list:
        return [message.get_dict() for message in messages]

    def append_message(self, content, direct=False, by_assistant=False) -> None:

        # while price_calculate(self.get_messages_list(self.indirect_messages)) > self.indirect_message_price:
        #     self.messages.remove(self.indirect_messages[0])
        #     self.indirect_messages.pop(0)
        
        # if price_calculate(self.get_messages_list(self.messages)) > self.max_price:
        #     self.reset_session()
        #     raise Exception("Price too high, reset session")

        role = "assistant" if by_assistant else "user"

        msg = self.new_message(role, content, direct_to_me=direct)

        # pure_msg = [msg.get_dict()]
        # if price_calculate(pure_msg) > 0.2:
        #     return
        
        if not direct:
            self.indirect_messages.append(msg)
        self.messages.append(msg)            

    def append_message_from_assistant(self, content) -> None:
        self.append_message(content, by_assistant=True, direct=True)

    def append_message_from_user(self, content, direct=False) -> None:
        self.append_message(content, direct=direct)

    # def get_overall_price(self) -> float:
    #     return price_calculate(self.get_messages_list(self.messages))
    
    def send_feedback(self) -> str:
        try:
            ret = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                # model="gpt-3.5-turbo",
                max_tokens=300,
                temperature=0.7,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.1,
                messages=self.get_messages_list(self.messages)
            )
        except Exception as e:
            self.reset_session()
            raise e
        self.append_message_from_assistant(ret.choices[0].message.content.strip())
        # price = price_calculate_by_token(ret.usage.total_tokens)
        # return ret.choices[0].message.content, price
        return ret.choices[0].message.content.strip()
    
class OpenAIGPT3(OpenAIChatGPT):
    def __init__(self, prompt_template = None) -> None:
        super().__init__(prompt_template)
    
    def send_feedback(self) -> str:

        messages = self.get_messages_list(self.messages)
        prompt = ""
        for message in messages:
            if message["role"] == "user":
                prompt += f"\nInput: {message['content']}"
            else:
                prompt += f"{message['content']}\n"
        try:
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.1
            )
        except Exception as e:
            self.reset_session()
            raise e
        self.append_message_from_assistant(response["choices"][0]["text"].strip())
        return response["choices"][0]["text"].strip()
