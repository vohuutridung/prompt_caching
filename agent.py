from crewai import Agent, Task, Crew, Process, LLM
import os
from dotenv import load_dotenv
from p_caching import retrieve_from_cache, store_to_cache
load_dotenv()

class CrewAgent:
    def __init__(self, model=''):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.model_id = os.getenv('MODEL_ID')
        self.llm = LLM(model=self.model_id, api_key=self.gemini_api_key,)

        self.agent = self.build_agent()
        self.crew = Crew(
            agents=[self.agent],
            tasks=[self.build_task()],
            process=Process.sequential,
            verbose=False,
        )

    def build_agent(self):
        agent = Agent(
            role='Gia sư',
            goal='Giải thích và trình bày kiến thức dễ hiểu cho học sinh',
            backstory='Bạn luôn muốn giúp đỡ các học sinh trong việc học tập, truyền đạt các kiến thức như một người thầy tâm huyết.',
            tools=[],
            llm=self.llm,
        )
        return agent

    def build_task(self):
        task = Task(
            description='Nhận câu hỏi sau: {question}, đua ra câu trả lời dễ hiểu, dành cho học sinh để nắm bắt kiến thức.',
            expected_output='Sử dụng từ ngữ cơ bản, ví dụ minh họa và giọng điệu vui vẻ để giải thích rõ ràng.',
            agent=self.agent,
        )
        return task
    
    def work(self, question):
        hit = retrieve_from_cache(question)
        if hit:
            print('Data successfully retrieved from cache!')
            return hit['result']
        
        result = self.crew.kickoff(inputs={'question': question})
        store_to_cache(question, result.raw)
        print('Data successfully stored to cache!')
        return result.raw


