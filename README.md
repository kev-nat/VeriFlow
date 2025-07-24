# Hierarchical Multi-Agent System for Supply Chain Anomaly Detection
This repository contains the source code and analysis for a novel hierarchical multi-agent framework designed for complex, real-time anomaly detection in supply chain management.

## Situation
The increasing complexity of global supply chains necessitates advanced systems for real-time risk assessment. While Large Language Models (LLMs) show promise for this task, their monolithic nature and high operational costs present significant barriers to scalable, real-world deployment.

## Objectives
- Create a robust, hierarchical multi-agent system for accurate, real-time anomaly detection.
- Investigate whether efficient, locally-deployable Small Language Models (SLMs) can serve as a high-performance, cost-effective alternative to proprietary APIs.
- Build a system that is easily adaptable to different business domains and requirements through a centralized configuration.

## Approaches
- **Hierarchical Multi-Agent Design:** A supervisor agent orchestrates a team of six specialized worker agents, each focused on a specific domain (e.g., Geolocation, Logistics, Human Factors), ensuring focused and efficient analysis.
- **Hybrid AI Architecture (The "Scaffolding Effect"):** The system separates deterministic logic from generative reasoning. A reliable, code-based rules engine handles precise anomaly flagging and scoring, allowing the SLM agents to focus on their core strength: summarizing the findings and synthesizing recommendations.
- **Config-Driven Modularity:** All system logic including agent responsibilities, anomaly rules, scoring weights, and model selection—is controlled via a single config file, making the entire framework highly adaptable.

## Impacts
- The architecture's deterministic core was validated with 100% accuracy in both Anomaly Detection and Risk Level classification across all tested models.
- The benchmark revealed that modern SLMs are highly competitive in generative quality. The best-balanced SLM (Gemma2) achieved an Agent Reason Similarity score of 0.761, closely approaching the GPT-4o Mini baseline of 0.802.
- Most critically, the hybrid SLM configuration resulted in a >50% reduction in processing time and a 50% reduction in operational API costs compared to a fully proprietary setup. This makes the system an economically viable and highly attractive solution for scalable, real-world deployment.
