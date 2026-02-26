from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully:\n\n"
    "1. **Extract Information:** Only extract details that match: {parse_description}.\n"
    "2. **No Extra Content:** Do not include explanations or extra comments.\n"
    "3. **Empty Response:** If no match is found, return an empty string ('').\n"
    "4. **Direct Data Only:** Output only the requested data, nothing else."
)

try:
    model = OllamaLLM(model="llama3.1")
except Exception as e:
    print(f"Error loading Ollama model: {e}")
    model = None 

def parse_with_ollama(dom_chunks, parse_description):
    if not model:
        return "Error: Ollama model is not available."

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):
        print(f"Processing chunk {i} / {len(dom_chunks)}")

        try:
            response = chain.invoke({"dom_content": chunk, "parse_description": parse_description})
            print(f"Response from chunk {i}: {response}")
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            response = "Error occurred"

        parsed_results.append(response)

    return "\n".join(parsed_results)
