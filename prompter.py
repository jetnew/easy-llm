import json
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

def llm(messages, model="gpt-4-turbo-preview", json_mode=False):
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    system = [{"role": "system", "content": "You are a helpful assistant designed to output JSON."}] if json_mode else []
    response = OpenAI().chat.completions.create(
        model=model,
        messages=system+messages,
        max_tokens=4096,
        **kwargs
    )
    response = response.choices[0].message.content.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    if json_mode:
        response = json.loads(response)
    return response


prompt = {
    "objective": "As a top recruiter, you are tasked to evaluate candidate profiles to judge if they are worth interviewing.",
    "inputs": [
        {
            "input": "Job",
            "field": "job"
        },
        {
            "input": "Profile",
            "field": "profile"
        }
    ],
    "outputs": [
        {
            "header": "Candidate's name",
            "field": "candidate_name",
            "instruction": "Please extract the candidate's name from the profile. Clean the name by removing any special characters or numbers.",
        },
        {
            "header": "Disqualification by lack of experience",
            "field": "disqualified",
            "instruction": "If the candidate has less or more than the acceptable years of experience, disqualify the candidate.",
            "format": "Y/N",
            "workings": [
                {
                    "header": "Job description years of experience",
                    "field": "job_years",
                    "instruction": "Extract the exact number of years of experience referencing only the job description.",
                },
                {
                    "header": "Acceptable years of experience",
                    "field": "acceptable_years",
                    "instruction": "Determine the acceptable range of years of experience for the job, i.e. +/- 1 years.",
                },
                {
                    "header": "Candidate profile years of experience",
                    "field": "candidate_years",
                    "instruction": "Extract the exact number of years of experience referencing only the candidate profile.",
                },
                {
                    "header": "Candidate years of experience within acceptable range",
                    "field": "within_range",
                    "instruction": "Determine if the candidate's years of experience falls within the acceptable range.",
                    "format": "Y/N"
                },
            ],
        },
        {
            "header": "Criteria 1: Relevance and length of past sales work experience in startups or leading SaaS companies",
            "field": "criteria_1_score",
            "instruction": "Assign a score from 0 to 3 based on 2 traits: Past sales work experience and Critical view of large corporation experience. ",
            "format": "0-3",
            "workings": [
                {
                    "header": "Trait 1: Past sales work experience",
                    "field": "trait_1",
                    "instruction": "Evaluate the candidate's experience in startups and leading sales-driven SaaS companies with evidence from the candidate's work history."
                },
                {
                    "header": "Trait 2: Critical view of large corporation experience",
                    "field": "trait_2",
                    "instruction": "Evaluate the candidate's experience with a critical view of large corporations with more than 1000 employees, except for companies that are recognized leaders in the SaaS industry."
                },
                {
                    "header": "Criteria 1: Overall evaluation based on trait 1 and trait 2",
                    "field": "criteria_1_evaluation",
                    "instruction": "Evaluate the candidate with concrete evidence from the candidate's professional history. Provide a thorough and fair evaluation based on specific examples."
                },
            ]
        }
    ]
}

def format_prompt(prompt):
    formatted_prompt = ""
    objective = f"# Objective\n<objective>\n{prompt['objective']}\n</objective>\n"

    fields = ""
    for i, output in enumerate(prompt['outputs']):
        fields += f"\n# {i+1}. {output['header']} - \"{output['field']}\"\n"
        fields += f"<task-{i+1}>\n"
        fields += f"{output['instruction']}\n"
        fields += f"Format: {output['format']}\n" if "format" in output else ""
        if "workings" in output:
            for j, working in enumerate(output['workings']):
                fields += f"\n## {i+1}.{j+1}. {working['header']} - \"{working['field']}\"\n"
                fields += f"<task-{i+1}.{j+1}>\n"
                fields += f"- {working['instruction']}\n"
                fields += f"Format: {working['format']}\n" if "format" in working else ""
                fields += f"</task-{i+1}.{j+1}>\n"
        fields += f"</task-{i+1}>\n\n"

    inputs = []
    for i, inp in enumerate(prompt['inputs']):
        inp = f"# {inp['input']}\n<{inp['field']}>\n{{{inp['field']}}}\n</{inp['field']}>\n"
        inputs.append(inp)
    inputs = '\n'.join(inputs) + '\n'

    format = "# Output Format\n"
    format += "<output-format>\n"
    for i, output in enumerate(prompt['outputs']):
        if "workings" in output:
            for j, working in enumerate(output['workings']):
                format += f"{i+1}.{j+1}. {working['field']}: \n"
        format += f"{i+1}. {output['field']}: \n"
    format += "</output-format>"

    formatted_prompt = objective + fields + inputs + format
    return formatted_prompt


if __name__ == "__main__":
    job = "Software Engineer, Google, 6-7 years experience."
    profile = "Jane Doe 🐼 is a software engineer with 5 years of experience."
    prompt = format_prompt(prompt).format(job=job, profile=profile)
    print(prompt)
    response = llm([{"role": "user", "content": prompt}])
    print("\n\nResponse:")
    print(response)