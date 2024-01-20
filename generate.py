from typing import Any
from llama_cpp import Llama
import time

class GenerateModel:
    ###
        # model_list = ["model/seallms/seallm-7b-chat.q4_k_s.gguf",
        #  "model/vinallama/vinallama-2.7b-chat_q5_0.gguf",
        #  "model/vinallama/vinallama-7b-chat_q5_0.gguf"]
    ###
    def __init__(self,
                 model_path: str = 'model/vinallama/vinallama-7b-chat_q5_0.gguf',
                 prompt:str=None #SEALLM, VINALLAMA
                 ) :
        
        print("---------------------------------------------------------------------")
        print("-------------------------Loading the LLM-----------------------------")
        print("---------------------------------------------------------------------")

        self.llm = Llama(model_path=model_path, 
            n_ctx=2048,
            n_batch=512,
            use_mlock=True,
            n_threads=8,
            verbose=False)
        
        print("---> Load LLM sucessfull.")
        if prompt == 'SEALLM':
            self.prompt = """Dưới đây là một hướng dẫn mô tả một nhiệm vụ. Viết phản hồi hoàn thành yêu cầu một cách thích hợp.
            Đưa ra câu trả lời thích hợp cho câu hỏi sau: 
            Câu hỏi: "{question}"
            Bạn chỉ cần sử dụng thông tin trong ngữ cảnh sau đây, bạn không sử dụng thông tin bên ngoài.
            Bạn cần trả lời bằng tiếng Việt và câu trả lời dài:
            Ngữ cảnh: "{context}"
            Câu trả lời là:"""  

        elif prompt == 'VINALLAMA':
            self.prompt = """<|im_start|>system
            Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
            <|im_end|>
            <|im_start|>user
            Hãy trả lời câu hỏi dưới đây bằng cách sử dụng thông tin có trong ngữ cảnh mà tôi cung cấp, tuyệt đối không thêm bất kì thông tin nào bên ngoài.
            Nếu ngữ cảnh không chứa câu trả lời hãy trả về "Tôi không có thông tin về câu hỏi này".
            Câu hỏi: "{question}"
            Ngữ cảnh: "{context}"
            <|im_end|>
            <|im_start|>assistant
            """

            self.prompt_no_context = """<|im_start|>system
            Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
            <|im_end|>
            <|im_start|>user
            {question}
            <|im_end|>
            <|im_start|>assistant
            """
    def __call__(self, question, context = None, temperature=0) -> Any:
        if context != None:
            input_prompt = self.prompt.format_map(
                {
                "question": question,
                "context": context
                }
            )
        else:
            input_prompt = self.prompt_no_context.format_map(
                {
                "question": question,
                }
            )
        start = time.time()
        output = self.llm(
            input_prompt,
            temperature=temperature,
            max_tokens=1024,
            stop=["Q:", "\n\n", "\n\n \n\n", "\n \n"],
            echo=False
        )
        answer = output['choices'][0]['text']
        finish_reason = output['choices'][0]['finish_reason']

        if answer == "Tôi không có thông tin về câu hỏi này":
            print("--> Do not find the context. Then answet with no context.\n\n")
            output = self.llm(
                self.prompt_no_context.format_map(
                {
                "question": question,
                }
                ),
                temperature=temperature,
                max_tokens=1024,
                stop=["Q:", "\n\n", "\n\n \n\n", "\n \n"],
                echo=False
            )

        end = time.time()
        answer = output['choices'][0]['text']
        finish_reason = output['choices'][0]['finish_reason']
        return answer, finish_reason, round(end - start, 2)

    
if __name__ == "__main__":
    generator = GenerateModel(prompt='VINALLAMA')
    answer = generator(question="Giới thiệu về đại học Bách khoa Hà Nội?", context=None)
    print(answer)