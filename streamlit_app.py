import streamlit as st
from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity # for fuzzy match
import numpy as np
import json

# Show title and description.
st.title("📄 RBM Template Generation Demo")
st.write(
    "Input a topic and GPT will provide a recommended outreach template for David @ Roundtable using public RBM Site Date and Optimation & DAPR case studies!"
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Fuzzy Matching Functions
def get_embedding(text, model="text-embedding-3-small"):
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)
    text = str(text)
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding
    
def compute_cosine_similarity(base_string, string_list, list_embeddings):
    model = "text-embedding-3-small"
    base_embedding = get_embedding(base_string, model)
    
    # Convert embeddings to numpy arrays
    base_embedding = np.array(base_embedding).reshape(1, -1)
    
    # Compute cosine similarity between the base string and all other strings
    similarity_scores = cosine_similarity(base_embedding, list_embeddings).flatten()

    return similarity_scores

def fuzzy_match(base_string, string_list, list_embeddings):
    '''
    How it works:
    Loading the Model: The script fetches the get_openai_embedding function fetches the embedding for a given text.
    Encoding Strings: The base string and each string in the list are encoded into high-dimensional vectors using the model.
    Cosine Similarity Computation: The cosine similarity between the base string vector and each vector in the string list is computed.
    Results Display: The similarity scores are printed for each string in the list relative to the base string. The script also identifies the string with the highest similarity score.
    '''
    output = ""
    # check whether the item exists exactly
    if base_string in string_list:
        output = base_string
    else:
        similarity_scores = compute_cosine_similarity(base_string, string_list, list_embeddings)
    
        # To find the most similar string
        most_similar_index = np.argmax(similarity_scores)
        output = string_list[most_similar_index]

        if similarity_scores[most_similar_index] < 0.4:
            st.write(f"Warning: not a perfect match, consider adding a more relevant case study. Similarity score: {similarity_scores[most_similar_index]}.") 
        
    return output, most_similar_index

def generate_embeddings_case_studies(df):
    # df["combined_text"] = df["case study"].astype(str) + " " + df["applications"] + " " + df["content"].astype(str)
    df["embeddings"] = df["combined_text"].apply(get_embedding)
    return df

def retrieve_case_study(query, df):
    output, index = fuzzy_match(query, df['case study'].tolist(), df["embeddings"].tolist())
    return output, index 


def template_generation(BD_doc, input, df_case_studies):
    
    # Retrieve the best case study
    case_study, index = retrieve_case_study(input, df_case_studies)
    st.write(f"Selected case study: {df_case_studies.iloc[index]['read more']}")

    # LinkedIn templates
    template_connection =  '''
    Hi %%first_name%%, I was interested by your work at %%company%%, and I came across a case study you may find interesting about {brief description of case study} for {brief description of client}. Happy to share more in case its relevant.
    '''
    template_message = '''
    Hi %%first_name%%,
    
    Thanks for connecting! As I mentioned earlier, I wanted to share this case study on {brief description of the case study}, which might be relevant to you or a colleague: {link}.
    
    For context, I founded Roundtable - a platform to help companies with external R&D. I thought Re:Build’s work in industrial process optimization might be of interest to you. They {briefly describe capabilities and key value propositions for that focus}.
    
    Would love to hear your thoughts!
    
    Best,
    David Chataway
    Founder of Roundtable, a free platform for R&D (www.roundtablehub.com)
    '''

    # FUNCTION TO REFINE THE INPUT INFORMATION 
    client = OpenAI(
        api_key=openai_api_key,
    )
    # Assemble the prompt
    system_prompt = '''
    ## Persona
    You are a technical business development assistant.

    ## Task
    You will receive a broad strategy doc, a case study and a user request as to which topic or market to focus on. Your must answer with the information in the strategy doc that pertains to that user request. Then make sure to augment that information by brainstorming:
    - a brief description of case study,
    - a brief description of the relevance of the case study,
    - a brief description of the outcome/value proposition of the case study,
    - an outline of Rebuild's capabilities and key value propositions for that market

    ## Output
    You should reply with all the relevant information, along with the addutional information requested.
    '''

    model_type = "gpt-4o-mini" 

    completions = client.chat.completions.create(
      model= model_type ,
      messages=[
        {"role": "system", 
         "content": system_prompt
        },
        {
          "role": "user",
          "content": '***Strategy Doc***: """' + str(BD_doc) + '"""\n\n***Case Study***: """' + str(df_case_studies.iloc[index]['combined_text']) + '"""' + '"""\n\n***User Request***: """' + str(input) + '"""'
        }
      ],
      temperature=0.2,
      max_tokens=4096,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    relevant_information = completions.choices[0].message.content

    # FUNCTION TO CREATE THE TEMPLATES
    client = OpenAI(
        api_key=openai_api_key,
    )
    # Assemble the prompt
    system_prompt = '''
    ## Persona
    You are a technical business development assistant.

    ## Task
    You will receive information about a company's marketing plan and services, a case study and LinkedIn message templates. You must complete the LinkedIn message templates by incorporating the information provided and following the general templates.

    ## Instructions
    - Be brief, and provide descriptions of the case study in 7 words or less.
    - Don't mention ReBuild's company name in the connection message.
    - Avoid corporate jargon and sounding robotic.
    - Fill out the templates in a way that seems natural and flows.
    - If the case study isn't exactly in the focus area, acknowledge that but explain the relevance.
    - You can be technical, your audience will be R&D professionals.

    ## Output
    You should reply with the completed LinkedIn messages in a JSON format.
    '''

    model_type = "gpt-4o-mini" 

    completions = client.chat.completions.create(
      model= model_type ,
      messages=[
        {"role": "system", 
         "content": system_prompt
        },
        {
          "role": "user",
          "content": '***Company Marketing Plan and Services***: """' + str(relevant_information) + '"""\n\n***Case Study***: """' + str(df_case_studies.iloc[index]['combined_text']) + '"""' + '"""\n\n***Case Study link***: """' + str(df_case_studies.iloc[index]['read more']) + '"""' + '"""\n\n***Focus***: """' + str(input) + '"""' + '"""\n\n***Connection_Template***: """' + str(template_connection) + '"""' + '"""\n\n***Message_Template***: """' + str(template_message) + '"""'
        }
      ],
      temperature=0.2,
      max_tokens=4096,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    
    return completions.choices[0].message.content



# Load the excel list and re-write the embeddings
df_case_studies = pd.read_excel('RBM_case_studies.xlsx', index_col=0)
df_case_studies = generate_embeddings_case_studies(df_case_studies)

# Load the BD document
BD_doc = '''
Industrial Process Optimization
________________________________________
1. Executive Summary
Re:Build Manufacturing’s industrial process optimization service uses advanced R&D techniques—from process modeling and simulation to data analytics, automation, and integration—to help industrial clients reduce waste, lower costs, and improve operational throughput. By leveraging a combination of cutting‐edge control systems, tailored automation, real‐time data acquisition, and continuous improvement methodologies, Re:Build delivers custom solutions that address both immediate bottlenecks and long-term growth challenges.
Key technical capabilities include:
•	Process simulation and pilot testing to validate process improvements.
•	High-speed data analytics using feedback systems (e.g., Model Free Adaptive Controllers, OEE metrics).
•	Integrated automation solutions from partial work cell upgrades to full, custom-tailored factory lines.
•	Robust systems integration including advanced sensor networks, digitization platforms, and embedded machine learning for predictive maintenance.
The campaign will target industrial sectors with heavy process optimization needs—specifically companies in chemical manufacturing, pharmaceuticals and biotech, industrial food & beverage processing, automotive parts manufacturing, aerospace/defense, energy processing, and advanced materials/composites. These industries exhibit sizable growth, demand technical R&D support, and are clustered within the U.S. and Canada.
________________________________________
2. Re:Build Manufacturing’s Competitive Advantage
•	Deep domain expertise and comprehensive R&D capabilities: From concept modeling to turnkey automation solutions, Re:Build’s engineers bring advanced simulation, process control, and integrated automation across a wide range of industrial sectors.
•	Custom, scalable solutions: Rather than off-the-shelf products, every engagement is analyzed for risk and economic impact to create a custom production roadmap that meets immediate client needs while positioning them for growth.
•	U.S.-based, secure manufacturing ecosystem: With facilities and design expertise distributed across the United States, Re:Build is well positioned to support reshoring initiatives and ensure IP security, improved supply chain stability, and regulatory adherence.
•	Continuous improvement & lean manufacturing: Emphasizing innovative design changes, rapid prototyping, and iterative improvements ensures clients enjoy ongoing cost savings and process reliability.
________________________________________
3. Market Segmentation and Opportunity Assessment
Below are seven key markets where Re:Build’s industrial process optimization R&D services can unlock significant value. Each market is broken down into segments, with an assessment of market attractiveness, geographical clusters, customer trends, innovation ecosystems, manufacturing challenges, and Re:Build’s tailored value proposition. Example target accounts are also provided for each segment.
________________________________________
Market 1: Chemical Manufacturing Process Optimization
A. Market Attractiveness
•	Market Size: The U.S. chemical industry generates revenues in the range of US$800+ billion annually. Specific segments (catalysts, specialty chemicals) are growing steadily at 3–5% per year.
•	Growth: Increasing regulatory pressures and the shift toward sustainability drive investments in R&D to improve production efficiency and reduce hazardous emissions.
•	Dynamics & Buyer Power: Highly competitive, with large chemical conglomerates wielding strong negotiating leverage; smaller specialty chemical manufacturers are increasingly receptive to bespoke R&D collaborations.
B. Geographical Clusters
•	Primary clusters in the Gulf Coast (Texas, Louisiana), Midwest, and Northeast due to the presence of major refining and chemical processing hubs.
C. Customer Segments
Segment 1.1: Continuous Process Optimization
•	Trends and Innovations: Adoption of real-time analytics, advanced sensor integrations, and digital twins to monitor and adjust process parameters.
•	Innovation Ecosystem: Federally funded research on green chemistry and process intensification; active partnerships with technical startups focused on process control.
•	Manufacturing Challenges: Maintaining consistent catalyst performance, safety regulatory compliance, and energy efficiency.
•	R&D Services Value Proposition: Customized simulation and pilot testing to optimize catalysts, reduce energy consumption, and increase throughput.
•	Target Examples:
o	Small: A regional specialty chemical manufacturer (e.g., a biotech-oriented catalyst producer)
o	Large: A multinational like Dow Chemical’s specialized division
Segment 1.2: Batch Process Improvement
•	Trends and Innovations: Innovations in batch mathematical modeling, process standardization, and automated quality control.
•	Ecosystem: Collaborative efforts with academic research centers and industry consortia focusing on batch processing efficiency.
•	Challenges: Variable production yields, consistency in quality, and regulatory tracking for each batch.
•	Value Proposition: Provide robust data analytics and adaptive control methodologies to standardize batch outputs and reduce lead times.
•	Target Examples:
o	Small: A regional chemical processing lab
o	Large: BASF manufacturing facilities (within U.S. operations)
________________________________________
Market 2: Pharmaceutical & Biotech Manufacturing
A. Market Attractiveness
•	Market Size: The U.S. pharmaceutical manufacturing market is worth over US$400 billion, with contract manufacturing growing at 6–8% per year.
•	Growth: Transitioning from batch to continuous manufacturing paradigms and increasing R&D outsourcing fuel demand for advanced process optimization.
•	Dynamics & Negotiating Power: High regulatory scrutiny strengthens buyer demands for quality assurance; however, smaller biotech firms and CMOs actively seek innovation partners.
B. Geographical Clusters
•	Clusters located in New Jersey, Massachusetts, and the Mid-Atlantic region where biotech hubs and pharmaceutical clusters coexist.
C. Customer Segments
Segment 2.1: Active Pharmaceutical Ingredient (API) Production
•	Trends and Innovations: Shift toward continuous processing and micro-reactor technologies; automation in sterile processing.
•	Ecosystem: Federally funded projects (e.g., FDA initiatives on continuous manufacturing) and dynamic startup ecosystems in biotech innovation.
•	Challenges: Stringent quality control, scalable process reliability, and compliance with cGMP standards.
•	Value Proposition: Deliver process simulations coupled with precision automation that minimizes contamination risks while ensuring production consistency.
•	Target Examples:
o	Small: A regional CMO focused on niche APIs
o	Large: Large contract manufacturers such as Lonza’s U.S. operations
Segment 2.2: Biotech Fermentation & Cell Culture Processes
•	Trends and Innovations: Integration of real-time monitoring and adaptive control systems for bioreactors; advanced data modeling for process variability.
•	Ecosystem: A mix of federally supported research and agile startups; emphasis on bio-process intensification.
•	Challenges: Balancing sensitivity of bioprocesses with process efficiency, and scale-up from lab to pilot production.
•	Value Proposition: Custom-engineered process control platforms that integrate sensor data, analytics, and lean automation to safeguard process integrity.
•	Target Examples:
o	Small: Early-stage biotech companies and local bio-incubators
o	Large: Major pharmaceutical companies with in-house R&D looking to augment production lines
________________________________________
Market 3: Industrial Food & Beverage Processing (Non-Consumer Focus)
A. Market Attractiveness
•	Market Size: The industrial segment of food & beverage manufacturing (processing for bulk, institutional, or food service supply) accounts for over US$250 billion across North America.
•	Growth: Driven by efficiency mandates and regulatory compliance, growth in automation is estimated at 5%+ annually.
•	Dynamics & Buyer Power: Moderate buyer power with a mix of large consolidated processors and agile mid-sized companies open to external R&D innovation.
B. Geographical Clusters
•	Strong presence in the Midwest, Southeast, and along the U.S. coasts where large-scale processing plants are located.
C. Customer Segments
Segment 3.1: Process Automation & Sanitization
•	Trends and Innovations: Adoption of robotics for non-contact processing to minimize contamination risks and innovations in inline sensor technologies for quality assurance.
•	Ecosystem: Collaborative research with government agencies on food safety and lean manufacturing; growing startup activity in automation.
•	Challenges: Balancing throughput with strict sanitary and safety regulations; retrofitting older plants with modern tech.
•	Value Proposition: End-to-end process assessments, customized automation integration, and real-time data acquisition tailored for food safety and efficiency.
•	Target Examples:
o	Small: A regional bulk food processing plant.
o	Large: Major integrated food processors with industrial-scale operations.
Segment 3.2: Packaging and Material Handling Automation
•	Trends and Innovations: Integration of vision inspection systems and custom machine design for resilient operations.
•	Ecosystem: Industry collaborations and technology incubators focusing on reducing waste and improving packaging throughput.
•	Challenges: Maintaining packaging quality and coordinating multiple production lines.
•	Value Proposition: Process modeling, custom systems integration, and automated quality checks to ensure efficiency and compliance.
•	Target Examples:
o	Small: Local co-packers.
o	Large: Well-capitalized food ingredient manufacturers.
________________________________________
Market 4: Automotive Parts Manufacturing & Assembly
A. Market Attractiveness
•	Market Size: The U.S. auto parts market surpasses US$900 billion in total value, with parts manufacturing representing a significant subsegment growing about 4–6% annually.
•	Growth: Demand for flexible, low-volume production solutions and enhanced process automation drives innovation in R&D.
•	Dynamics & Buyer Power: Consolidated tier-1 suppliers often have strong bargaining power; niche manufacturers seek more specialized process improvements.
B. Geographical Clusters
•	Concentration in the Midwest (especially Michigan and Ohio) and parts hubs in the Southern U.S. where automotive supply chains are dense.
C. Customer Segments
Segment 4.1: Stamping & Assembly Line Optimization
•	Trends and Innovations: Implementation of robotic pick-and-place systems, lean manufacturing principles, and flexible automation to address variable production volumes.
•	Ecosystem: Active collaborations with industry research labs and lean manufacturing programs; startup innovations in automation platforms are prevalent.
•	Challenges: High variability in production runs, quality control under tight tolerances, and capital constraints in smaller facilities.
•	Value Proposition: Use of state-of-the-art process simulation and custom machine design to improve line efficiency, reduce rework, and minimize downtime.
•	Target Examples:
o	Small: A local component fabricator.
o	Large: Major tier-1 supplier such as Magna’s U.S.-based units.
Segment 4.2: Robotics Integration for Advanced Assembly
•	Trends and Innovations: Increased adoption of robotics in precision assembly, especially with emerging digital control systems and real-time adaptive algorithms.
•	Ecosystem: Partnerships with technical universities and federally funded robotics research centers.
•	Challenges: Integrating legacy systems with advanced digital control platforms and ensuring seamless human–robot collaboration.
•	Value Proposition: Provide integration services that bridge innovative digital controls with established manufacturing infrastructure, reducing retrofit risks.
•	Target Examples:
o	Small: Specialized robotics integrators serving the automotive subsector.
o	Large: Large-scale OEM suppliers investing in digital transformation.
________________________________________
Market 5: Aerospace & Defense Manufacturing
A. Market Attractiveness
•	Market Size: The U.S. aerospace and defense manufacturing sector is valued in the hundreds of billions, with process optimization opportunities estimated to support multi-billion-dollar efficiency gains.
•	Growth: Driven by technological innovation and digital transformation initiatives, anticipated growth in R&D spending is approximately 5–7% annually.
•	Dynamics & Buyer Power: Buyer expectations are very high, with rigorous quality and compliance standards; suppliers face stiff competitive pressures yet benefit from significant project sizes.
B. Geographical Clusters
•	Clusters in the Southeast (e.g., Alabama and Georgia), Pacific Northwest, and areas surrounding major aerospace hubs in California and the Northeast.
C. Customer Segments
Segment 5.1: Composite Materials & Precision Assembly
•	Trends and Innovations: Incorporation of digital twins and simulation-based R&D to ensure component integrity; increased precision in automated assembly has been a focus.
•	Ecosystem: Close ties to federally funded research agencies (e.g., NASA, DoD labs) and strong involvement from established aerospace R&D startups.
•	Challenges: Tight tolerances, regulatory adherence, and demanding certification processes.
•	Value Proposition: Offer end-to-end R&D services including simulation, custom automated testing, and iterative improvements to reduce waste and improve compliance.
•	Target Examples:
o	Small: A boutique composite component manufacturer.
o	Large: Aerospace tier-1 suppliers such as those servicing Boeing or Lockheed Martin.
Segment 5.2: Advanced System Integration for Defense Applications
•	Trends and Innovations: Emphasis on cybersecurity in control systems, integration of sensor networks, and real-time process monitoring.
•	Ecosystem: Government-funded R&D projects and active participation from defense technology startups.
•	Challenges: Balancing the need for rapid innovation with strict security, compliance, and reliability requirements.
•	Value Proposition: Provide integrated systems engineering that combines robust security protocols with adaptive manufacturing process automation.
•	Target Examples:
o	Small: A regional defense contractor.
o	Large: Large defense manufacturing integrators.
________________________________________
Market 6: Energy Production & Industrial Processing
A. Market Attractiveness
•	Market Size: The energy sector, including refining and petrochemical processing, is a multi-hundred-billion-dollar market in North America.
•	Growth: Sustainability and safety initiatives drive a 3–5% annual growth in energy process optimization R&D.
•	Dynamics & Buyer Power: Large refineries hold significant power, while smaller processors seek high-ROI R&D solutions.
B. Geographical Clusters
•	Key regions: Gulf Coast, Midcontinent, parts of Canada with refining and petrochemical assets.
C. Customer Segments
Segment 6.1: Refinery Process Automation
•	Trends: Increased use of smart sensors, predictive analytics, and integrated control systems.
•	Ecosystem: Collaborations between federal energy bodies and startups focusing on green tech.
•	Challenges: Integrating legacy infrastructure, risk management, and safety compliance.
•	Value Proposition: Scalable simulations and safety automation using Re:Build’s expertise.
•	Target Examples:
o	Small: Regional refinery modernization.
o	Large: Major energy conglomerates (e.g., Chevron).
Segment 6.2: Petrochemical Production Efficiency
•	Trends: Advanced control systems and adaptive algorithms to optimize yield and emissions.
•	Ecosystem: R&D partnerships and environmental grants support process digitization.
•	Challenges: Balancing efficiency with environmental regulations, high capital risks.
•	Value Proposition: Custom technology upgrades and iterative simulations.
•	Target Examples:
o	Small: Specialized petrochemical firm.
o	Large: Large multinational refining operations.
________________________________________
Market 7: Advanced Materials & Composites Manufacturing
A. Market Attractiveness
•	Market Size: The market for advanced materials and composites manufacturing is valued at US$150–200 billion globally, with notable segments in the U.S. and Canada.
•	Growth: Expected annual growth of 7–9%, driven by demand in aerospace, automotive, and industrial sectors.
•	Dynamics & Buyer Power: Niche manufacturers and startups drive R&D investment, while large companies seek innovative supplier partnerships.
B. Geographical Clusters
•	Key regions: Northeast, California, and Midwestern hubs supporting advanced composites research.
C. Customer Segments
Segment 7.1: Custom Composite Manufacturing
•	Trends: Rapid prototyping, additive manufacturing, and material informatics transforming production.
•	Ecosystem: Startups and federal research initiatives fueling innovation.
•	Challenges: High R&D costs and integrating novel materials into legacy processes.
•	Value Proposition: R&D services, prototype development, advanced testing, and digital optimization.
•	Target Examples:
o	Small: Startups in novel composite formulations.
o	Large: Industrial materials companies like Owens Corning.
Segment 7.2: Electronic Materials Optimization
•	Trends: Machine learning in production processes for high-performance materials.
•	Ecosystem: Partnerships with federal agencies and material innovation accelerators.
•	Challenges: Scaling lab innovations while ensuring material precision.
•	Value Proposition: Process modeling and integrated automation for quality and rapid scale-up.
•	Target Examples:
o	Small: Specialized electronic materials startups.
o	Large: Long-standing industrial suppliers in advanced applications.
________________________________________
Ranked Opportunity Assessment
Based on market size, growth potential, R&D funding, and Re:Build’s capabilities, the following ranking reflects relative opportunity:
1.	Chemical Manufacturing Process Optimization
2.	Pharmaceutical & Biotech Manufacturing
3.	Energy Production & Industrial Processing
4.	Aerospace & Defense Manufacturing
5.	Automotive Parts Manufacturing & Assembly
6.	Industrial Food & Beverage Processing (Non-Consumer)
7.	Advanced Materials & Composites Manufacturing
________________________________________
Conclusion
This business development plan outlines a tailored outbound marketing strategy focusing on industrial process optimization in key U.S. and Canadian manufacturing sectors. By highlighting Re:Build Manufacturing’s competitive advantages, targeting specific market segments with detailed R&D value propositions, and strategically approaching prospects through relaxed yet technical messaging, the campaign is well-positioned to create meaningful partnerships and drive long-term growth.

'''



# Ask the user for a question via `st.text_area`.
question = st.text_area(
    "Now give me details about a topic or commercial use case.",
    placeholder="Can you give me a short summary of the market or company you want to focus on?"
)

if question:
    
    output = template_generation(BD_doc, question, df_case_studies)
    # st.json(output)
    st.write(output)
