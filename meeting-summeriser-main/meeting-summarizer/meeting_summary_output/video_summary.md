# Meeting Summary: AI-Based Marine Animal Rescue System

**1. Overall Summary:**

The meeting presented the "Straw Hats" team's proposal for an AI-powered system to detect and rescue injured or stranded marine animals.  The system utilizes drone, satellite, and CCTV footage, along with crowdsourced reports and environmental data, to identify distressed animals, alert authorities, and guide rescue efforts.  The team detailed their technical approach, including data acquisition, processing, AI models (YOLOv9, TCN, Transformer), and the tech stack (Node.js, FastAPI, ReactJS, etc.).  The presentation heavily relied on text-based slides outlining the problem, solution, unique features, and implementation details.  A process flowchart was a key visual element, demonstrating the system's workflow.

**2. Key Discussion Points:**

* Problem Statement: The significant delay in detecting injured or stranded marine animals, leading to missed rescue opportunities and animal deaths due to a lack of real-time monitoring.
* Proposed Solution: An AI-based detection and alert system analyzing drone, satellite, and CCTV footage to identify distressed animals and send alerts with GPS coordinates to relevant authorities.
* Unique Features: Crowdsourced reporting app, guided citizen rescue support, marine species recognition and data logging for research, environmental context integration, heatmaps of incident zones, and live dashboards for public awareness.
* Technical Approach: Multimodal data acquisition (drone, satellite, CCTV, sonar, acoustic sensors, environmental APIs), data preprocessing with OpenCV, AI model application (YOLOv9 for object detection, TCN for behavior analysis, Transformer for data fusion), and rescue route planning using GIS and environmental data.
* Tech Stack: Backend (Node.js, FastAPI), Frontend (ReactJS, Mapbox), AI models (PyTorch, YOLOv9, TCN, Transformer), Data Acquisition (DJI SDK, Copernicus, RadarSat, OpenWeatherMap), Cloud tools (AWS EC2, AWS S3, PostgreSQL).
* Implementation Plan: A phased approach starting with multimodal data collection and iterative model training based on logged data.


**3. Important Visuals:**

* **Initial Slides (0-40s):**  These slides primarily contained text describing the project title ("KRITHOATHON 3.0"), team name ("Team Straw Hats"), problem statement, and a project ID ("PSID: KR305").  They lacked diagrams or charts.
* **"What Makes Us Unique" Slides (50-120s):** A series of slides displayed a numbered list outlining six key features of the proposed system.  These were text-based, without visual aids.
* **Abstract Slides (140-210s):**  These slides contained only the text of the abstract describing the system.
* **Implementation Plan Slides (230-400s):**  These slides presented a step-by-step implementation plan, again mainly text-based.
* **Process Flowchart (300-500s):** This was the most significant visual.  A flowchart displayed on a smartphone/tablet screen illustrated the data flow and processing steps within the system, including data acquisition from multiple sources, AI-based analysis, risk assessment, and alert generation.  It showcased the system's complexity and interconnectivity.  The flowchart included boxes for "Multi-Source Data," "Distress Probability + Species ID," "Biodiversity Logging," "Signal Filtering," "Emergency Alerts," "Rescue Route Planning," and many other key processes.
* **Tech Stack Slide (530-590s):** A slide outlining the technologies used, categorized into Backend & APIs, Frontend & Visualization, Machine Learning & AI Models, Data Acquisition, and Other Tools.


**4. Image Time-stamps:**

Significant visual frames are present at approximately the following timestamps: 0, 50, 140, 300, 530, and 600 seconds.  The flowchart (300-500s) is particularly important.

**5. Action Items (if mentioned):**

No specific action items were identified with deadlines or assigned individuals.  The presentation focused on outlining the project proposal.

**6. Decisions Made (if any):**

No specific decisions were identified during the meeting.  The purpose was to present a project proposal.
