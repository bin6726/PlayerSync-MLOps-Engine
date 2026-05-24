Markdown
# PlayerSync AI — Production MLOps Engine 🚀

An enterprise-level User Behavioral Segmentation and Machine Learning interface built with Python and Streamlit, fully containerized using Docker, and deployed on an AWS EC2 cloud instance.

---

## 🏗️ System Architecture

The application is structured as a modern production MLOps pipeline, ensuring environment isolation, automated dependency tracking, and global accessibility:

```text
[ Local PC / WinSCP ] 
        │  (Secure Code & Model Sync via SFTP)
        ▼
[ AWS EC2 Instance (Mumbai Region) ]
        │
        ├──► [ AWS Security Group Firewall ] ──► (Inbound Route: Port 8501)
        │
        └──► [ Docker Container Engine ]
                 └──► [ playersync-container ] ──► (Python 3.10-slim + Streamlit App)
Key Technical Components:
Machine Learning Engine: Utilizes unsupervised machine learning (K-Means Clustering) to ingest real-time user engagement metrics and dynamically segment players into behavioral archetypes (e.g., The Whale Spenders).

Containerization (Docker): Wrapped in a light, optimized python:3.10-slim container blueprint to handle environment setup, file paths, and application isolation deterministically.

Cloud Infrastructure (AWS): Hosted on an Amazon Web Services EC2 cloud node behind customized inbound network routing security groups.

🛠️ Tech Stack & Tooling
Language: Python 3.10

Frontend Dashboard: Streamlit

Containerization Engine: Docker

Cloud Platform: Amazon Web Services (AWS EC2)

Deployment/Terminal Tools: PuTTY & WinSCP

Version Control: Git & GitHub

📂 Project Repository Structure
Plaintext
PlayerSync_AI/
├── app.py                  # Main Streamlit dashboard & ML inference app
├── Dockerfile              # Docker structural image instructions
├── requirements.txt        # Production Python library versions
├── cluster_model.pkl       # Serialized Machine Learning clustering model
└── scaler.pkl              # Serialized ML feature scaling weights
🚀 Local & Cloud Deployment Instructions
1. Build the Docker Image
To compile the application environment from the customized blueprint, navigate to your root project folder and execute:

Bash
docker build -t playersync-app .
2. Launch the Isolated Container
To initialize and spin up the production application container mapping out to port 8501, run:

Bash
docker run -d -p 8501:8501 --name playersync-container playersync-app
3. Verify Container Status
Check container health and trace actively running runtime environments via:

Bash
docker ps
🌐 Network Configuration Note
To access the live front-end application via the AWS Public IP string (http://<AWS_PUBLIC_IP>:8501), ensure your AWS Security Group Inbound Rules are explicitly configured to pass communication payloads across a Custom TCP pointer targeting Port 8501 originating from source flag 0.0.0.0/0 (Anywhere-IPv4).
