"""
title: Storm Wiki Pipeline
author: zabirauf
date: 2024-07-22
version: 1.0
license: MIT
description: Research about a topic and create a wikipedia like content containing summary and information with varied perspective.
requirements: knowledge_storm
environment_variables: OPENAI_API_KEY, YOU_API_KEY, REGULAR_MODEL_NAME, SMART_MODEL_NAME
"""

import tempfile
import os
from typing import List, Union, Generator, Iterator
from openai import OpenAI


from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.rm import YouRM
from pydantic import BaseModel, Field
from utils.pipelines.main import get_last_user_message
import json

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = ""
        YOU_API_KEY: str = ""
        REGULAR_MODEL_NAME: str = "gpt-4o-mini"
        SMART_MODEL_NAME: str = "gpt-4o"


    def __init__(self):
        super().__init__()

        self.type = "manifold"
        self.name = "Storm Wiki Pipeline"
        self.valves = self.Valves(
            **{
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                "YOU_API_KEY": os.getenv("YOU_API_KEY", ""),
                "REGULAR_MODEL_NAME": os.getenv("REGULAR_MODEL_NAME", "gpt-4o-mini"),
                "SMART_MODEL_NAME": os.getenv("SMART_MODEL_NAME", "gpt-4o"),
            },
        )

        self.pipelines = [
            { "id": "storm-wiki-researcher", "name": "Storm-Wiki-Researcher" }
        ]

    def research_topic(
                self,
                topic: str,
            ) -> str:
        """
        Research about a topic and create a wikipedia like content containing summary and information with varied perspective.

        :param topic: The topic user want to research about
        :return: The wikipedia like content containing summary and more detailed information about the topic
        """
        if self.valves.OPENAI_API_KEY == "":
            return "OpenAPI Key not set, ask the user to set it up."
        elif self.valves.YOU_API_KEY == "":
            return "You.com API Key not set, ask the user to set it up."
        else:

            lm_configs = STORMWikiLMConfigs()

            openai_kwargs = {
                'api_key': self.valves.OPENAI_API_KEY,
                'temperature': 1.0,
                'top_p': 0.9,
            }
            regular_model_name = self.valves.REGULAR_MODEL_NAME
            smart_model_name = self.valves.SMART_MODEL_NAME
            conv_simulator_lm = OpenAIModel(model=regular_model_name, max_tokens=500, **openai_kwargs)
            question_asker_lm = OpenAIModel(model=regular_model_name, max_tokens=500, **openai_kwargs)
            outline_gen_lm = OpenAIModel(model=smart_model_name, max_tokens=400, **openai_kwargs)
            article_gen_lm = OpenAIModel(model=smart_model_name, max_tokens=700, **openai_kwargs)
            article_polish_lm = OpenAIModel(model=smart_model_name, max_tokens=4000, **openai_kwargs)

            lm_configs.set_conv_simulator_lm(conv_simulator_lm)
            lm_configs.set_question_asker_lm(question_asker_lm)
            lm_configs.set_outline_gen_lm(outline_gen_lm)
            lm_configs.set_article_gen_lm(article_gen_lm)
            lm_configs.set_article_polish_lm(article_polish_lm)

            # Create a temporary directory for output
            self.temp_output_dir = tempfile.mkdtemp()
            print(f"Created temporary output directory: {self.temp_output_dir}")

            # Update engine_args with the temporary directory
            engine_args = STORMWikiRunnerArguments(
                output_dir=self.temp_output_dir,
                max_conv_turn=3,
                max_perspective=3,
                search_top_k=3,
                max_thread_num=3,
            )

            research_module = YouRM(ydc_api_key=self.valves.YOU_API_KEY, k=5)
            runner = STORMWikiRunner(engine_args, lm_configs, research_module)
            print("Starting runner.run() with topic:", topic)
            runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=True,
            )
            print("runner.run() completed")

            print("Starting runner.post_run()")
            runner.post_run()
            print("runner.post_run() completed")

            print("Starting runner.summary()")
            runner.summary()
            print("runner.summary() completed")

            # Read the polished article from the output directory
            article_path = os.path.join(runner.article_output_dir, 'storm_gen_article_polished.txt')
            print(f"Attempting to read polished article from: {article_path}")
            print(f"Checking if article exists at path: {article_path}")
            if os.path.exists(article_path):
                print("Article found. Reading content...")
                with open(article_path, 'r', encoding='utf-8') as f:
                    polished_article = f.read()

                # Read and parse the citations JSON file
                try:
                    citations_path = os.path.join(runner.article_output_dir, 'url_to_info.json')
                    if os.path.exists(citations_path):
                        with open(citations_path, 'r', encoding='utf-8') as f:
                            citations_data = json.load(f)
                        
                        # Generate markdown for citations
                        citations_markdown = self.generate_citations_markdown(citations_data)
                        
                        # Append citations to the polished article
                        polished_article += "\n\n" + citations_markdown
                    else:
                        print("Citations file not found.")
                except Exception as err:
                    print(f"Error getting citations file: {str(err)}")

                # End of Selection
                print(f"Article content read. Length: {len(polished_article)} characters")
                return polished_article
            else:
                print("Error: Polished article not found at the specified path.")
                return "Error: Polished article not found."

    def generate_citations_markdown(self, citations_data: dict) -> str:
        citations_markdown = "## Citations\n\n"
        # Create a list of tuples containing citation number and URL
        citations_list = [(int(citations_data['url_to_unified_index'].get(url, '0')), url) for url in citations_data['url_to_info'].keys()]
        # Sort the list based on citation number
        citations_list.sort(key=lambda x: x[0])
        
        for citation_number, url in citations_list:
            info = citations_data['url_to_info'][url]
            title = info.get('title', 'Untitled')
            description = info.get('description', 'No description available')
            snippets = info.get('snippets', [])
            
            citations_markdown += f"### [{citation_number}] [{title}]({url})\n\n"
            citations_markdown += f"{description}\n\n"
            if snippets:
                citations_markdown += "**Relevant Snippets:**\n\n"
                for snippet in snippets[:3]:  # Limit to 3 snippets
                    citations_markdown += f"- {snippet}\n\n"
            citations_markdown += "---\n\n"
        
        return citations_markdown

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    
    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Get the last user message
        last_user_message = get_last_user_message(messages)

        # Count the number of user messages
        user_message_count = sum(1 for message in messages if message["role"] == "user")

        # Check if this is the first or second user message
        if user_message_count <= 1:
            # Use OpenAI to extract the topic from the user message
            try:
                client = OpenAI(api_key=self.valves.OPENAI_API_KEY)
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Extract the main research topic from the user's message. Respond with only the topic, nothing else."},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=500
                )
                
                topic = response.choices[0].message.content.strip()
                
                # Now use the extracted topic to generate the article
                return self.research_topic(topic)
            except Exception as e:
                return f"Error extracting topic: {str(e)}"
        else:
            return "I'm sorry, I can't chat about previous research."