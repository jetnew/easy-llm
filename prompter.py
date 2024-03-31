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
        temperature=0,
        **kwargs
    )
    response = response.choices[0].message.content.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    if json_mode:
        response = json.loads(response)
    return response

job = """COMPANY DESCRIPTION
Hi. We’re OFX, a global provider of online, international payment services. We solve the complexity
of moving money and enable better decisions. Headquartered in Sydney with offices worldwide,
we’re a customer-focused business that is all about inspiring customer confidence.
At OFX, you’ll have the opportunity to reach beyond your role and function across disciplines. Make
use of your diverse skill set at a business that values your expertise and turn your potential into
reality.
JOB DESCRIPTION
We are hiring a Business Development Manager for Singapore, who will be an integral part of the
Singapore OFX team. You will play an important part expanding revenue streams by qualifying and
acquiring new leads across the Mid-Size Business Segment. You will focus on solution-based sales, to
educate prospective clients on OFX's service and products. You will collaborate with other members
of the sales and dealing team to successfully secure prospective clients and build business
relationships.
What you will be doing:
• Intelligently research, identify and build a strong pipeline of leads and opportunities.
• Maintain strong pre-sales activities through your own network and direct prospecting.
• Present and explain OFX's value proposition, products and services to the prospective
clients.
• Meet and exceed revenue targets as an individual; help overall team in achieving their
targets.
• Collaborate with OFX's broader team to stay updated with new and existing compliance and
legal regulations, product advancements and distribution channels.
• Actively contribute and consult on longer-term strategic sales direction in Asia. Support
strategic initiatives which will enable the Asia team to deliver on market expansion and
revenue targets.
• Manage data for new and prospective clients in Salesforce, maintain tracking and recording
activity.
QUALIFICATIONS
• University Degree preferred
Knowledge, skills, experience:
• Skilled in B2B Sales with at least 5-6 years’ experience in selling financial services to
Singapore SMEs.
• A proven track record of achieving and exceeding your revenue targets, in past.
• FX experience, familiarity with FX spot, Forward Contracts products and hedging knowledge
is advantageous.
• A positive attitude and mindset, with the drive to succeed and willingness to cold call.
• Ability to identify customer’s needs and map it to the solutions that OFX offers.
• Ability to navigate challenging customer interactions, as well as internal stakeholder
management.
• Possess excellent communication and interpersonal skills.
• Be a solid team player, collaborative and self-motivated.
• Languages – Excellent spoken and written English.
• Possess integrity, reliability, and strong sales ethics.
• Salesforce experience is preferred but not essential.
This is a hybrid position. Hybrid employees can alternate time between both remote and office.
Employees in hybrid roles are expected to work from the office at least once a week or more (based
on business needs).
ADDITIONAL INFORMATION
What it's like working at OFX
We’re OFXers because we want to make a difference. We see challenges as opportunities and we’re
not afraid to roll up our sleeves to get stuff done. We’re committed to making things easier for our
clients, pushing boundaries and continuing to move with the times so that we can continue to
inspire confidence every day and through every transaction.
We operate as one team, cross-functionally and globally to drive outcomes that deliver excellence
for our customers. We're curious self-starters who love learning and sharing our knowledge with
others. We embrace change and use our initiative and resilience to overcome challenges.
• Always keep learning. We offer LinkedIn learning programs, which all OFXers have access to.
We offer a variety of other learning programs and host an annual Open Day to encourage
cross functional and soft skill learning
• Giving back, we encourage OFXers to give back to causes and communities that are
important to them. We celebrate this with an annual volunteer day, that OFXers can use
together or individually
• We promote an environment of reward and recognition, OFXers are encouraged to
celebrate their peers’ effort, technical expertise or support through a range of channels
and awards
• Our Good Vibes employee-led committees organize events to keep our employees engaged
inside and outside the office. Whether it’s participating in our weekly yoga class (now also
on Zoom), office happy hours, end of year celebrations. Our team wants you to feel
welcomed!"""

profile = """Rayner Kwok
  3rd degree connection3rd
Business Development | FinTech | Ant Group


Business Development ManagerBusiness Development Manager
WorldFirstWorldFirst
Jan 2018 - Present · 6 yrs 3 mosJan 2018 - Present · 6 yrs 3 mos
  
World First - Ant Group
Helping people and businesses move money around the world most efficiently

Scope
- Optimize channel for all Inbound clients throughout their customer journey
- Outbound sales engaging MDs, CFOs, FMs, Operation Managers
- Partner with Corporate Secretaries and Business Consultants to provide Multi Currency accounts and Payment solutions


Business Development ExecutiveBusiness Development Executive
Hesed & Emet HoldingsHesed & Emet Holdings
Apr 2016 - Dec 2017 · 1 yr 9 m


AssociateAssociate
IAM AdvisoryIAM Advisory
Aug 2015 - Nov 2015 · 4 mos

Financial AdvisorFinancial Advisor
GREAT EASTERNGREAT EASTERN
Nov 2013 - Jul 2015 · 1 yr 9 mos

NBS Alumni Affairs, NTUNBS Alumni Affairs, NTU
Bachelor of Business Administration (BBA), Tourism and Hospitality ManagementBachelor of Business Administration (BBA), Tourism and Hospitality Management
2012 - 20152012 - 2015
Grade: (Hon)


Financial AnalysisFinancial Analysis
TeamworkTeamwork
1 endorsement1 endorsement
BankingBanking
FinanceFinance
1 endorsement1 endorsement
SalesSales
1 endorsement"""


scoring_prompt = {
    "objective": "As a top 1% recruiter, you must evaluate a range of profiles to decide if they are worth interviewing. Understand that many profiles may not be extensively detailed, and some strong candidates might have less filled-out profiles. Despite this, it's crucial to discern the quality of profiles using subtle clues and human-like inferences. Rely on your deep experience and heuristic methods to make final judgments, even when the information seems insufficient. If you are making inferences, it must be reasonably and logically drawn from a fact or group of facts established in the candidate's profile. You cannot make mere conjecture or speculation that does not properly flow from the facts in the candidate profile. Your goal is to identify the potential in candidates beyond the completeness of their profiles, leveraging your expertise to differentiate between good and bad candidates effectively. You are to assess candidates based on three key criteria. Allocate up to 3 points for Criteria 1, up to 5 total points for Criteria 2, and up to 2 points for Criteria 3. Avoid a rigid scoring system for each criterion; instead, adopt a holistic approach, making inferences and using your judgment to evaluate the overall impression of the candidate. Aim to select candidates in the top 10 percentile, rigorously filtering out those who don't meet the standard. Be stringent with scoring; reserve high scores for candidates who distinctly excel in these criteria. Generous scoring could lead to hiring unsuitable candidates, jeopardizing the growth trajectory of our rapidly expanding startup. After scoring each criteria, provide a paragraph that justifies the points awarded. This justification should reflect how the candidate's overall profile and potential align with the criteria. Once all scoring is done, provide a holistic overview where you must highlight both pros and cons of the candidate. If a candidate is disqualified due to falling out of the acceptance range of experience, this must be emphasized as a major con. Then give the final score and final recommendation on whether to interview. Lastly, suggest a list of three specific areas or questions to explore during the interview process. These should be aimed at addressing or gaining further insight into the identified cons of the candidate's profile. The goal is to clarify these points and assess how the candidate could potentially overcome or mitigate these weaknesses. You must suggest exactly three areas, no more, no less.",
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
            "instruction": "It is crucial to mark candidates whose years of relevant experience exceeds or falls short of the expected years of experience range as 'Disqualified'. This step is critical and must be done, as hiring candidates outside the expected range could adversely affect the business. Therefore, ensure rigorous compliance with the disqualification criteria. It is imperative to ensure that the process of automatic disqualification for candidates falling outside the expected experience range does not influence the overall scoring system. The scoring system should remain unaffected by disqualification criteria to serve educational and reference purposes",
            "format": "Y/N",
            "outputs": [
                {
                    "header": "Job description years of experience",
                    "field": "job_years",
                    "instruction": "Analyze the provided job description to extract the clearly specified minimum and maximum years of professional experience required. Record the job description's experience range stated.",
                    "format": "Min years: <min_years>, Max years: <max_years>"
                },
                {
                    "header": "Expected years of experience",
                    "field": "expected_years",
                    "instruction": "Define an expected experience range that accommodates candidates who are slightly less or more experienced than the stipulated requirements.",
                    "format": "Min years: <min_years>, Max years: <max_years>"
                },
                {
                    "header": "Candidate profile years of experience",
                    "field": "candidate_years",
                    "instruction": "Initiate the evaluation by calculating each candidate's total years of professional experience. Apply discernment to omit internships and avoid double-counting periods of concurrent employment. Document the calculated years of professional experience for each candidate.",
                    "format": "<years> years <months> months"
                },
                {
                    "header": "Candidate years of experience exceed or falls short of expected years of experience",
                    "field": "candidate_exceeds_or_falls_short_expected_years",
                    "instruction": "State if the candidate's years of experience exceeds or falls short of the expected years of experience defined by 'expected_years'. Inspect the candidate's years of experience and the expected years of experience, then determine if the candidate's years of experience exceeds or falls short of the expected years of experience.",
                    "format": "Y/N"
                },
                # {
                #     "header": "Disqualification Reasoning",
                #     "field": "disqualification_reasoning",
                #     "instruction": "Provide a detailed working of whether a candidate is disqualified based on his years of relevant experience and the job description's stated required years of experience.",
                #     "format": "Answer every question: What's the job description's required YOE? What's an acceptable YOE range? What's the candidate's YOE? Does the candidate's YOE exceed, meet, or fall short of the acceptable YOE range? What is the final decision?"
                # }
            ],
        },
        {
            "header": "Criteria 1: Relevance and Length of past Sales work Experience in start-ups or leading SaaS companies",
            "field": "criteria_1_score",
            "instruction": "Assess a candidate's work history by looking at the following 2 traits. Assign scores to candidates only when their work history meets the criteria established by one or more of the traits. Do not score for any reasons outside of these 2 predefined traits. Only 3 maximum points can be assigned for criteria 1.",
            "format": "0-3",
            "outputs": [
                {
                    "header": "Trait 1: Past Sales Work Experience Evaluation",
                    "field": "criteria_1_trait_1",
                    "instruction": "It is essential for you to conduct careful and comprehensive research on the background of the companies that the candidate has worked in. Score Criteria: There are 2 types of work experiences you can give high scores for. Type 1 - Start-up experience: Assign high points to candidates with experience in startups which qualifies in either of the following two ways: 1. The startup has obtained funding from venture capitalists, indicating financial endorsement and market potential. 2. The startup/company is widely acknowledged for its rapid growth within the same industry as the hiring company as per the job description, demonstrating success and relevance in the market. Type 2 - Leading Sales-Driven SaaS Companies: Allocate high points to candidates who have worked in sales roles at SaaS (Software as a Service) companies known for their market dominance and robust growth. These should be companies that are recognized as leading entities in the SaaS sector, with a strong track record of market leadership and expansion. In the scoring process, award points exclusively to candidates who have demonstrated significant success and active involvement in startups or leading SaaS companies. This highlights their proven capacity to thrive in competitive, growth-centric environments. You must not assign high points for work experiences in companies that fall outside of these two categories: startups, or leading software companies. This restriction ensures the scoring accurately mirrors the candidate's aptitude for a high-performance startup ecosystem, focusing solely on their proven track record in these specific, dynamic business environments. Justify scores by giving evidence from the candidate's work history that shows their experience in (i) startups or (ii) leading SaaS companies. You must also explain why these companies fall under (i) startups or (ii) leading SaaS Companies.",
                    "format": "Paragraph: Detailed Evidence, Explanation, and Justification"
                },
                {
                    "header": "Trait 2: Critical View of Large Corporation Experience",
                    "field": "criteria_1_trait_2",
                    "instruction": "Candidates from large corporations with more than 1000 employees are not favored, except in rare cases where they have worked in companies that are recognized leaders in the SaaS industry. Emphasis is placed on the agility, innovative mindset, and hands-on experience typically found in smaller business environments.",
                    "format": "Paragraph: Detailed Evidence, Explanation, and Justification"
                },
                {
                    "header": "Criteria 1: Overall evaluation based on trait 1 and trait 2",
                    "field": "criteria_1_evaluation",
                    "instruction": "For each trait assessed, ensure that the scoring is supported by concrete evidence from the candidate's professional history. Do not give high scores without explicit and overwhelming evidence of the traits. This methodical approach ensures a thorough and fair evaluation of each candidate's suitability based on specific, predefined factors.",
                    "format": "Paragraph: Detailed Evidence, Explanation, and Justification"
                },
            ]
        },
        {
            "header": "Criteria 2: Sales Competency",
            "field": "criteria_2_score",
            "instruction": "Assess the candidate's proficiency in full-cycle outbound sales as an individual contributor. This evaluation should focus on three equally important traits, with none being overlooked during the assessment process: 1. Prospecting Skills: Evaluate the candidate's demonstrated ability to independently prospect for deals, including their effectiveness in cold calling and emailing. Look for evidence of proactive lead generation and initial contact strategies. 2. Closing Skills: Assess the candidate's ability to close deals, emphasizing their presentation skills. This includes evaluating their technique in presenting products or services, handling objections, and securing commitments from prospects. 3. Quantifiable Success: Examine the candidate's track record in full-cycle outbound B2B sales, focusing on measurable outcomes and achievements. Look for clear evidence of their impact on sales growth, deal size, and customer acquisition within previous roles. Use the rating scale to assign a score that best represents your evaluation of each criterion. Very Bad (1 point): Significantly below expectations; major improvements needed. Somewhat Bad (2 points) : Below expectations; several areas need improvement. Okay (3 points) : Meets basic expectations; acceptable but not outstanding. Good (4 points): Above expectations; generally solid performance. Exceptional (5 points): Exceeds expectations; exceptional performance. Only 5 maximum points can be assigned in total for criteria 2. Evaluate candidates for their potential and suitability in sales based on your professional judgment. Approach each profile case by case, avoiding premature conclusions. Focus on the depth and impact of the information provided rather than the volume of content. Significant, impactful achievements or roles should be prioritized over a long list of less substantial experiences. You are allowed to assign a high score for criteria 2 based on your professional judgment, even in cases of less detailed profiles, under the premise that any gaps can be addressed and clarified in the interview stage. However, as there is less detail, you must be extremely confident about the broader context of the candidate's experiences and achievements before giving a high score.",
            "format": "0-5",
            "outputs": [
                {
                    "header": "Relevant Heuristics",
                    "field": "relevant_heuristics",
                    "instruction": "You are always required to apply expert heuristics below made by top 1% sales professionals to aid in the assessment. Adhere strictly to the specified list below. Indicate which heuristics were used to make your justification. Provide a clear linkage and detailed explanation showing how the candidate's profile aligns with or deviates from the heuristic used. List of heuristics: Nuanced understanding of sales roles: As a top recruiter with a nuanced and specialized understanding of sales roles, you recognize that proficiency in one sales category doesn't necessarily equate to competence in another. For example, an account manager who manages clients post-sale, will not know how to source and close clients from scratch. Seek to understand which area the candidate is most skilled in based on the following categories. 1. Sourcing Clients 2. Sourcing & Closing Clients 3. Closing Leads Brought by Marketing or Partnerships 4. Managing Clients After Won 5. Partnership 6. Leadership Pick one category which best describes the candidate. Use your inference to decide if the candidate has sufficient transferable skills based on all their past work experiences relevant to the job we are hiring for. Assessing Sales Achievements: When evaluating a candidate's sales experience, prioritize quantifiable achievements that reflect their capacity to exceed performance benchmarks. Examples of key performance indicators include: Exceeded sales targets, with a benchmark of achieving 100% or more. Exceed quota attainment, aiming for more than 100% achievement. Top performer and ranked 1 in my team. Contributing significantly to Annual Recurring Revenue (ARR), with a notable threshold of over $100,000. Achieving a high percentage of annual sales targets, such as 100% for fiscal year. Any awards, honors, or recognitions, as these indicate the candidate's recognition by peers or the industry, highlighting their expertise and success. Testimonials or feedback from colleagues, managers, or clients. Positive feedback can provide insights into the candidate's work ethic, impact, and interpersonal skills. These metrics should be used to gauge the candidate's effectiveness in their previous sales roles, indicating their potential for high performance in future sales positions. Scan for these specific figures and contexts to assess the candidate's track record of success and their ability to contribute substantially to sales goals. B2C vs B2B Sales experience: You must make an inference on whether the candidate's sales experience is B2B or B2C. B2B sales experience must be prioritized over B2C sales experience for roles that specifically require B2B sales skills. B2B sales often involve longer sales cycles, larger deal sizes, and more complex decision-making processes than B2C sales. In contexts where B2B sales skills are essential, B2C sales experience is not seen favorably. Priority for Software and Venture-Backed Startup sales Experience: Candidates with sales experience in (i) startups or (ii) leading sales led software companies should be given more credit. Sales experience in these sectors is more valued due to the fast-paced and challenging nature of the technology and startup world, where rapid growth and scalability are priorities. Consequently, past experience in companies outside of these specific sectors should be considered less relevant and given less weight in the assessment process. Focus on the alignment of the candidate's previous sales experience with the high-growth and innovative nature of our business to ensure their skills and experiences are directly applicable to our needs. Career Progression: Look for evidence of fast career advancement where a candidate has at least 1 promotion within the same company or upward movement across companies every 0-3 years. Any pattern of fast career progression suggests competence and success in their roles. Pay attention to implicit indicators of success and seniority, such as job titles that include 'Senior', 'Lead', or 'Head of'. These can suggest a higher level of responsibility and achievement, even in the absence of explicit details. Continuous Learning: Look for signs of ongoing professional development, like certifications, courses, or participation in industry events. This indicates a commitment to growth and staying current in their field. Quality of Education: Consider the reputation of the educational institutions the candidate attended, as attending well-regarded institutions can be a proxy for their ability to meet and maintain high standards.",
                    "Format": "List of Heuristics"
                },
                {
                    "header": "Heuristics Justification",
                    "field": "heuristics_justification",
                    "instruction": "Construct a very detailed justification that critically evaluates the candidate's sales competence. You must weigh all strengths and weaknesses about their sales competence according to the criteria before making a final judgment and score.",
                    "format": "Paragraph: Detailed Evidence, Explanation, and Justification"
                }
            ]
        },
        {
            "header": "Criteria 3: Fit for Specific Skills Required in Job Description",
            "field": "criteria_3_score",
            "instruction": "This assessment concentrates on two crucial traits that are of equal importance and should be thoroughly evaluated without bias. Only these specified traits should influence the scoring, avoiding consideration of unrelated factors. If there is no clear evidence of the traits required, and any inference of such traits seems too speculative or loosely connected, then a score of 0 should be assigned to reflect this lack of concrete evidence. The evaluation of traits should be precise and evidence-based. Avoid granting the candidate the benefit of the doubt unless there is a strong and justifiable reason. This approach ensures that the scoring remains objective and accurately reflects the candidate's proven traits in relation to the job's requirements. Each trait can be awarded a maximum of 1 point, totaling a possible 2 points.",
            "format": "0-2",
            "outputs": [
                {
                    "header": "Trait 1: Industry-Specific Knowledge, Skills, and Experience",
                    "field": "criteria_3_trait_1_score",
                    "instruction": "Evaluate the candidate's industry-specific knowledge, skills, and experience based on the job description.",
                    "format": "0-1",
                    "outputs": [
                        {
                            "header": "Industry Sector",
                            "field": "industry_sector",
                            "instruction": "Before evaluating the candidate, you must identify key terms and the industry sector. Identifying Industry sector. Define the industry sector targeted by the hiring job as per the job description. You must not refer to anything other than the job description to get the key terms. Follow these steps to ensure a detailed and precise definition: Industry: Start by identifying the broad category that the job role falls under based on the job description, such as Financial Services, Technology, or Healthcare. Sub-Sector: Within the identified industry, pinpoint the specific segment the job role is involved with based on the job description only. Examples include Investment Banking, Retail Banking, or FinTech in the Financial Services industry. This further refines the area of focus. Niche: Craft a concise, 10-word sentence that accurately describes the company's particular area of specialization within the sub-sector based on the job description. This should highlight the unique aspect of the company's operations in the market. Example: Industry: Financial Services, Sub-Sector: Financial Technology (FinTech), Niche: Online payment processing solutions. This definition ensures a comprehensive understanding of the job role's industry sector, aiding in the accurate alignment of a candidate's experience with the specific demands of the job.",
                            "format": "[<industry-sector>, <sub-sector>, <niche>]",
                        },
                        {
                            "header": "Industry Specific Knowledge, Skills and Experience Key Terms",
                            "field": "industry_specific_key_terms",
                            "instruction": "Identifying Key Terms. Candidates with direct experience in the hiring company's specific industry sector, as demonstrated by their knowledge, skills, or previous work experience, are often more successful in the role. Such candidates merit higher scores. To evaluate this, you must initially pinpoint the precise knowledge, skill, or experience in the specific industry sector required by the hiring job by analyzing the job description. Look for clear keywords, word combinations, or exact phrases that define the specialized industry knowledge, skills, or experience needed within the job description. For example: 'Experience selling Fraud, Anti-Money Laundering, or Brokerage Compliance Solutions' 'In-depth understanding and experience in selling solutions related to Enterprise Search, Log Analytics, Security, APM, and Cloud'. You must not consider general sales skills or experiences that are not directly linked to the specified industry, such as generic B2B sales experience or familiarity with Salesforce. The aim is only to ascertain the candidate's specialized expertise and experience in the targeted industry sector, and not general proficiency. Correct identification of at least 6 key terms are crucial as they directly relate to the specialized knowledge, skills, or experience required for the role. You must not refer to anything other than the job description to get the key terms. Record the specific keywords, word combinations, or exact phrases extracted from the job description only. Do not refer to the candidate profile.",
                            "format": "[<term-1>, ...]",
                        },
                        {
                            "header": "Trait 1: Overall evaluation",
                            "field": "criteria_3_trait_1_evaluation",
                            "instruction": "Scoring guidelines for Trait 1: Keyword Matching: Assess the candidate's LinkedIn profile for the presence of specific keywords, word combinations, or exact phrases identified and documented previously. The occurrence of these terms suggests the candidate's proficiency and direct involvement in the relevant industry. Synonym and Semantic Similarity Analysis: Extend the evaluation to include synonyms and semantically similar terms that closely match the identified keywords. This approach helps capture candidates with relevant experience described in different but closely related terms. You can only look for synonyms and semantically similar terms that are exceptionally close in meaning, avoiding any broad or loose correlations to maintain assessment accuracy. Company Affiliation as a Skill Indicator. Analyze the companies where the candidate has worked to assess their industry-specific skills, knowledge, or experience. This analysis involves researching these organizations to understand the candidate's level of involvement and the relevance of their experience to the hiring job's industry sector. When scoring based on company affiliation, prioritize the closeness of the candidate's previous companies to the niche focus within the industry sector definition above. These elements are crucial as they represent the specialized area within which the candidate must have expertise. Assign less weight to similarities in broader categories of overall industry and sub-sector alignment, as these are more general and less indicative of specialized knowledge and skills. Justify the weighting of different aspects of the industry definition in the scoring process. Educational and Certification Review: Review the candidate's educational background and certifications, especially those related to the hiring job's industry sector. Credentials that are closely related to the required industry skills and knowledge should be viewed favorably and indicate a higher level of qualification for the role. Award up to 1 point for the candidate's demonstrated depth and relevance of experience in these areas, ensuring an evaluation only of their industry-specific expertise.",
                            "format": "Paragraph: Detailed Evidence, Explanation, and Justification"
                        }
                    ]
                },
                {
                    "header": "Trait 2: Candidate's Experience with Target Customer Segments: ",
                    "field": "criteria_3_trait_2_score",
                    "instruction": "For this trait, you must ascertain the size of the companies that the candidate will target in their sales role, as specified in the job description. The goal is to match the candidate's experience with the customer segment they will be dealing with. Allocate up to 1 point based on the depth and relevance of the candidate's experience with the identified customer segment. A higher score should reflect a strong and direct match between the candidate's past sales engagements and the customer segment targeted in the sales role.",
                    "format": "0-1",
                    "outputs": [
                        {
                            "header": "Target Customer Segments",
                            "field": "target_customer_segments",
                            "instruction": "Identify Target Company Size: Review the job description to determine the company size the sales role is focused on. The description will typically specify the type of companies the salesperson will be expected to engage with. Categorize According to Defined Segments: For the evaluation process, it's important to categorize the target customer segments as outlined in the job description into one of the following defined segments: 1. Small and Medium Enterprises (SMEs): Characterized by having fewer than 250 employees and an annual revenue of less than $50 million. This segment includes smaller businesses that operate on a local or regional scale, often requiring tailored solutions and personalized sales approaches. 2. Mid-Market: This segment bridges the gap between SMEs and large enterprises, typically having between 250 and 1,000 employees and annual revenues ranging from $50 million to $1 billion. Mid-market companies often have more complex needs than SMEs but are more nimble than large enterprises. 3. Enterprise: Defined by more than 1,000 employees and annual revenues exceeding $1 billion. Enterprise customers are usually large multinational corporations with complex organizational structures and extensive procurement processes. Document the Target Segment: Clearly state which of the four customer segments (SMEs, Mid-Market, Enterprise) the job description aligns with. This identification will guide the evaluation of the candidate's experience and suitability for targeting the specified customer size.",
                        },
                        {
                            "header": "Trait 2: Overall evaluation",
                            "field": "criteria_3_trait_2_evaluation",
                            "instruction": "Evaluate candidate's Experience with target segment: Review the candidate's profile to identify evidence of their sales experience with customers that match the company size identified in the job description (SMEs, Mid-Market, or Enterprise). This assessment will highlight the candidate's capability and familiarity in dealing with the specified customer segment. Synonym and Semantic Similarity Search: Broaden the evaluation by including synonyms and semantically similar terms that accurately reflect the target customer segment. This approach ensures that relevant experiences are considered, even if they are described using slightly different terminology. It's crucial to maintain a strict criterion for similarity to avoid incorporating irrelevant experiences.",
                            "format": "Paragraph: Detailed Evidence, Explanation, and Justification"
                        }
                    ]
                },
                {
                    "header": "Criteria 3: Overall evaluation based on trait 1 and trait 2",
                    "field": "criteria_3_evaluation",
                    "instruction": "Evaluate the candidate with concrete evidence from the candidate's professional history. Provide a thorough and fair evaluation based on specific examples.",
                    "format": "Paragraph: Detailed Evidence, Explanation, and Justification"
                }
            ]
        },
        {
            "header": "Holistic Overview Final Score",
            "field": "holistic_overview_score",
            "instruction": "Provide a holistic overview final score based on the candidate's profile evaluation.",
            "format": "0-10",
            "outputs": [
                {
                    "header": "Holistic Overview Pros",
                    "field": "holistic_overview_pros",
                    "instruction": "List the pros of the candidate based on the holistic overview evaluation.",
                    "format": "Paragraph: Detailed Evidence, Explanation, and Justification"
                },
                {
                    "header": "Holistic Overview Cons",
                    "field": "holistic_overview_cons",
                    "instruction": "List the cons of the candidate based on the holistic overview evaluation.",
                    "format": "Paragraph: Detailed Evidence, Explanation, and Justification"
                }
            ]
        },
        {
            "header": "Interview Recommendation",
            "field": "interview_recommendation",
            "instruction": "Recommend whether the candidate should be interviewed based on the holistic overview final score.",
            "format": "Y/N",
            "outputs": [
                {
                    "header": "Interview Exploration Areas by Overview Cons",
                    "field": "interview_exploration_areas",
                    "instruction": "List the areas that need further exploration during the interview based on the holistic overview cons.",
                    "format": "[<area-1>, <area-2>, <area-3>]"
                }
            ]
        }
    ]
}

def format_objective(prompt):
    return f"# Objective\n<objective>\n{prompt['objective']}\n</objective>\n"

def format_fields(prompt, d=1, p=""):
    fields = ""
    for i, output in enumerate(prompt['outputs']):
        fields += f"\n{'#'*d} {p}{i+1}. {output['header']} - \"{output['field']}\"\n"
        fields += f"<task-{p}{i+1}>\n"
        fields += f"{output['instruction']}\n"
        fields += f"Format: {output['format']}\n" if "format" in output else ""
        if "outputs" in output:
            fields += format_fields(output, d+1, p=f"{p}{i+1}.")
        fields += f"</task-{p}{i+1}>\n"
    return fields

def format_inputs(prompt):
    inputs = []
    for i, inp in enumerate(prompt['inputs']):
        inp = f"# {inp['input']}\n<{inp['field']}>\n{{{inp['field']}}}\n</{inp['field']}>\n"
        inputs.append(inp)
    inputs = '\n' + '\n'.join(inputs) + '\n'
    return inputs

def format_outputs(prompt):
    format = "# Output Format\n"
    format += "<output-format>\n"
    def format_out(prompt, p=""):
        format = ""
        for i, output in enumerate(prompt['outputs']):
            if "outputs" in output:
                format += format_out(output, p=f"{p}{i+1}.")
            format += f"{p}{i+1}. {output['field']}: \n"
        return format
    format += format_out(prompt)
    format += "</output-format>"
    return format

def format_prompt(prompt):
    objective = format_objective(prompt)
    fields = format_fields(prompt)
    inputs = format_inputs(prompt)
    outputs = format_outputs(prompt)
    return objective + fields + inputs + outputs


if __name__ == "__main__":
    prompt = format_prompt(scoring_prompt).format(job=job, profile=profile)
    print(prompt)
    response = llm([{"role": "user", "content": prompt}])
    print("\n\nResponse:")
    print(response)