import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import PyPDF2
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Initialize Gemini Client
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)
MODEL_NAME = 'gemini-2.5-flash'

MANIM_RULES = """
You are a Manim Community Edition (v0.19) code generator for educational animations.

Convert the following description into a *single, self-contained Manim scene*.

STRICT RULES — follow every one:
 1. Start with from manim import *
 2. Define exactly ONE class called GeneratedScene(Scene):
 3. Set background: self.camera.background_color = "#0a1224" in construct()
 4. Use Text(...) for explanations (*never* use Tex for plain text).
   - *CRITICAL*: ALWAYS set width=config.frame_width - 2 on every Text() to prevent text from going off-screen.
   - Keep each individual Text() call to max ~80 chars. If the content is longer, split it across MULTIPLE Text objects shown in SEPARATE frames/animation steps.
   - Use font_size=28 for body text, font_size=40 for titles.
 5. Use MathTex(...) for LaTeX equations/formulas.
   - *CRITICAL LaTeX rules for MathTex:*
     - NEVER use \\text{} — it causes compilation errors. Use plain Text() objects beside the equation instead.
     - NEVER mix units or words inside MathTex. Keep MathTex ONLY for pure math: variables, operators, numbers.
     - For units like "kg", "m/s", "MeV", put them in a separate Text() positioned next to the equation.
     - Use ONLY basic LaTeX: ^, _, \\frac{}{}, \\sqrt{}, \\times, \\cdot, \\Delta, \\sum, \\int, \\vec{}, \\hat{}.
     - NEVER use \\textbf, \\textit, \\mathrm, \\mbox, \\hbox inside MathTex.
     - For long equations, use font_size=36 and break across lines using \\\\ inside an aligned environment.
     - Always test: if the LaTeX string has English words in it, those words should be in a separate Text(), NOT in MathTex().
 6. Use animations: Write, FadeIn, FadeOut, Create, Transform, ReplacementTransform
 7. Add self.wait(1) to self.wait(2) between sections so the viewer can read.
 8. Keep the total animation under 60 seconds (about 10-20 animation steps). Use more steps to present content clearly.
 9. Use colors: BLUE_C, YELLOW_C, GREEN_C, RED_C, WHITE, GREY_A for variety.
10. Position elements carefully — use .to_edge(), .shift(), .next_to() to avoid overlaps.
    - Keep a margin of at least 0.5 units from all screen edges.
    - Use VGroup(...).arrange(DOWN, buff=0.5) to stack text blocks vertically.
11. Clear the screen with self.play(FadeOut(*self.mobjects)) between major sections.
12. DO NOT use any external files, images, SVGs, or custom fonts.
13. DO NOT use Tex() — only Text() and MathTex().
14. Output ONLY valid Python code. No markdown fences, no comments outside the code, no explanations.
15. Structure each section as: Title → Key point (1-2 short lines) → Equation (if any) → Clear screen → Next section.
16. NEVER put a long paragraph in a single Text(). Break content into bite-sized pieces across multiple frames.
17. *CRITICAL*: Do NOT add ANY explanatory text, comments, or [instruction] tags after the code ends. Output ONLY the Python code itself.

REMEMBER: Output ONLY the Python code. No explanations before or after. No [instruction] tags. Just pure Python code.
"""

class Topics(BaseModel):
    topics: list[str]

class Subtopics(BaseModel):
    subtopics: list[str]

class Descriptions(BaseModel):
    descriptions: list[str]

def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def api_call_with_retry(func, *args, **kwargs):
    """Wrapper to execute an API call with automatic 2, 8, 16, 32s retries."""
    delays = [2, 8, 16, 32]
    last_exc = None
    for attempt, delay in enumerate(delays + [0]):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < len(delays):
                print(f"\nAPI Error ({type(e).__name__}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Failed completely after retries. Final error: {e}")
    if last_exc:
        raise last_exc

def _fetch_topics(pdf_text: str):
    prompt = f"Extract a comprehensive list of all specific concepts and detailed topics listed under every unit in the following JEE/NEET syllabus text. Do NOT just list the broad unit names (like 'Oscillations and Waves'). Instead, list the granular topics detailed under them (e.g., 'Oscillations and periodic motion: time period, frequency, displacement as a function of time', 'Simple harmonic motion (S.H.M.) and its equation', 'Wave motion, longitudinal and transverse waves', etc.). Return each of these detailed segments as a separate string element.\\n\\n{pdf_text[:100000]}"
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Topics,
            temperature=0.2,
        ),
    )
    if not response.parsed:
        raise ValueError("Response heavily filtered or invalid JSON. Response parsed is None.")
    return response.parsed.topics

def get_topics(pdf_text: str) -> list[str]:
    print("Extracting detailed topics using Gemini...")
    try:
        return api_call_with_retry(_fetch_topics, pdf_text)
    except:
        return []

def _fetch_subtopics(topic: str, num: int):
    prompt = f"For the physics/chemistry/math topic '{topic}', generate a list of {num} highly specific, visualizable subtopics that can be animated in Manim."
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Subtopics,
            temperature=0.7,
        ),
    )
    if not response.parsed:
        raise ValueError("Response heavily filtered or invalid JSON. Response parsed is None.")
    return response.parsed.subtopics

def get_subtopics(topic: str, num: int = 20) -> list[str]:
    try:
        return api_call_with_retry(_fetch_subtopics, topic, num)
    except Exception as e:
        print(f"Could not fetch subtopics for {topic}: {e}")
        return []

def _fetch_descriptions(subtopic: str, num_desc: int):
    prompt = f"""Generate {num_desc} diverse and detailed "Animation Blueprints" for the educational subtopic: '{subtopic}'.

**CRITICAL INSTRUCTION:**
These blueprints will serve as the strict input prompt to fine-tune an LLM. During actual usage, another LLM will generate this exact blueprint format to tell the fine-tuned model what to animate. 
To ensure consistency between training and future inference, each blueprint string must strictly follow this exact structural template without deviating:

**Scene Sequence**:
1. [Action 1 - e.g., Display the title at the top of the screen using Text]
2. [Action 2 - e.g., Write the main equation using MathTex on the left]
3. [Action 3 - e.g., Add Text explaining the variables next to the equation]
4. [Action 4 - e.g., Fade things out to clear the screen]
...

Keep the sequences realistic for an educational 10-20 step Manim animation. Describe specifically what text arrays to display and what equations to write.
Return {num_desc} distinct, creative blueprints as a JSON list of strings."""
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Descriptions,
            temperature=0.8,
        ),
    )
    
    # REMOVED: print(response.text) <--- This was printing all descriptions to the terminal, making it look like they were being skipped!
    
    if not response.parsed:
        raise ValueError("Response heavily filtered or invalid JSON. Response parsed is None.")
    return response.parsed.descriptions

def generate_descriptions(subtopic: str, num_desc: int = 15) -> list[str]:
    try:
        return api_call_with_retry(_fetch_descriptions, subtopic, num_desc)
    except:
        return []

def _fetch_manim_code(description: str):
    prompt = f"{MANIM_RULES}\nDescription: {description}"
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
        ),
    )
    code = response.text.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()

def process_description_task(description: str):
    """Worker function for threads to generate code for a single description."""
    try:
        code = api_call_with_retry(_fetch_manim_code, description)
        return {"description": description, "manim_code": code}
    except Exception as e:
        # FIXED: Catching the exception and printing it so it doesn't fail silently
        print(f"Failed to generate Manim code: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Dataset Generator for Manim")
    parser.add_argument("--pdf", default="jee_neet_syllabus.pdf", help="Path to syllabus PDF")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent api calls")
    parser.add_argument("--total", type=int, default=20000, help="Target pair count")
    args = parser.parse_args()

    # 1. Load or Extract Topics
    topics_file = "topics_cache.json"
    if os.path.exists(topics_file):
        with open(topics_file, "r") as f:
            topics = json.load(f)
        print(f"Loaded {len(topics)} topics from cache.")
    else:
        pdf_text = extract_text_from_pdf(args.pdf)
        topics = get_topics(pdf_text)
        with open(topics_file, "w") as f:
            json.dump(topics, f, indent=4)
        print(f"Extracted {len(topics)} topics.")

    # 2. Extract Subtopics using ThreadPool
    subtopics_file = "subtopics_cache.json"
    if os.path.exists(subtopics_file):
        with open(subtopics_file, "r") as f:
            subtopics = json.load(f)
        print(f"Loaded {len(subtopics)} subtopics from cache.")
    else:
        print("Generating subtopics using ThreadPoolExecutor...")
        subtopics = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_topic = {executor.submit(get_subtopics, t, 15): t for t in topics}
            for future in tqdm(as_completed(future_to_topic), total=len(topics), desc="Generating Subtopics"):
                subs = future.result()
                if subs:
                    subtopics.extend(subs)
                    with open(subtopics_file, "w") as f:
                        json.dump(subtopics, f, indent=4)

        print(f"Generated {len(subtopics)} subtopics.")

    # 3. Main Data Generation Loop - fully parallelized batches
    dataset_file = "dataset.jsonl"
    existing_pairs = 0
    generated_descriptions = set()
    if os.path.exists(dataset_file):
        with open(dataset_file, "r") as f:
            for line in f:
                data = json.loads(line)
                generated_descriptions.add(data["description"])
                existing_pairs += 1
    
    print(f"Already generated {existing_pairs} pairs. Target is {args.total}.")

    if existing_pairs >= args.total:
        print("Target reached. Exiting.")
        return

    pbar = tqdm(total=args.total, initial=existing_pairs, desc="Dataset Pairs Generated")
    
    # Process subtopics in batches equal to number of workers to prevent running out of memory
    batch_size = args.workers
    with ThreadPoolExecutor(max_workers=args.workers * 2) as executor:
        for i in range(0, len(subtopics), batch_size):
            if existing_pairs >= args.total:
                break
                
            batch_subs = subtopics[i : i + batch_size]
            
            # Step 1: Parallel fetch descriptions
            future_to_sub = {executor.submit(generate_descriptions, sub, 10): sub for sub in batch_subs}
            all_descriptions_for_batch = []
            
            for future in as_completed(future_to_sub):
                desc_list = future.result()
                if desc_list:
                    # Keep only non-duplicated descriptions
                    new_descs = [d for d in desc_list if d not in generated_descriptions]
                    all_descriptions_for_batch.extend(new_descs)
            
            if not all_descriptions_for_batch:
                continue

            # FIXED: Slice the array so we only request exactly as many API tasks as we need to reach the target.
            # This prevents wasting credits on tasks that would otherwise get abandoned when it breaks the loop.
            remaining_needed = args.total - existing_pairs
            descriptions_to_process = all_descriptions_for_batch[:remaining_needed]

            # Step 2: Parallel fetch manim code for ONLY the sliced descriptions
            future_to_desc = {executor.submit(process_description_task, d): d for d in descriptions_to_process}
            
            for future in as_completed(future_to_desc):
                result = future.result()
                if result:
                    generated_descriptions.add(result["description"])
                    existing_pairs += 1
                    pbar.update(1)
                    
                    with open(dataset_file, "a") as f:
                        f.write(json.dumps(result) + "\n")
            
            # We don't need a break statement inside the as_completed loop anymore, 
            # the slicing guarantees we won't exceed the target count.
            if existing_pairs >= args.total:
                break

    pbar.close()

if __name__ == "__main__":
    main()