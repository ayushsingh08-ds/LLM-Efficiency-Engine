#!/usr/bin/env python3
"""
Load test: Send 100 diverse prompts through the gateway.
Tracks request counts, costs, latency, quality, and provider distribution.
"""

import requests
import json
import time
import statistics
from collections import defaultdict
from typing import Optional

# Gateway URL
GATEWAY_URL = "http://localhost:8000"
GENERATE_ENDPOINT = f"{GATEWAY_URL}/generate"

# Diverse prompt templates for realistic variety
DIVERSE_PROMPTS = [
    # Short/simple queries
    "What is Python?",
    "Explain machine learning briefly",
    "Define microservices",
    "What is REST API?",
    "Explain caching",
    
    # Code snippets/requests
    "Write a function to find the nth Fibonacci number in Python",
    "Create a SQL query to find duplicate emails in a users table",
    "How do I reverse a string in JavaScript?",
    "Show me a simple React component example",
    "Write pseudocode for binary search",
    
    # Medium prompts
    "Explain the difference between monolithic and microservices architecture. Include pros and cons of each.",
    "What are the best practices for API design? List at least 5 key principles.",
    "Compare SQL and NoSQL databases. When would you use each?",
    "Describe the role of a load balancer in a distributed system.",
    "What is CI/CD and why is it important for software development?",
    
    # Long/complex prompts
    "Design a system to handle 1 million users simultaneously. Consider database design, caching strategy, load balancing, and monitoring.",
    "Explain the CAP theorem. How do distributed databases trade off consistency, availability, and partition tolerance? Provide real-world examples.",
    "Create a comprehensive disaster recovery plan for a mission-critical application. Include RTO, RPO, backup strategies, and failover mechanisms.",
    "Design an e-commerce platform that scales globally. Address payment processing, inventory management, user experience, and compliance.",
    "Build a recommendation engine architecture. Explain data collection, feature engineering, model selection, and real-time serving.",
    
    # Creative/narrative
    "Write a short story about an AI learning to understand human emotions",
    "Create a comedic dialogue between a programmer and a non-technical manager",
    "Imagine a future where code writes itself. Describe this scenario in 200 words",
    "Write a haiku about debugging",
    "Create marketing copy for a fictional cloud computing service",
    
    # Data science
    "What are the top 5 machine learning algorithms and when to use each?",
    "Explain the difference between supervised and unsupervised learning with examples",
    "How do you handle missing data in a dataset? List 5 techniques",
    "What is overfitting and how do you prevent it?",
    "Explain cross-validation and why it matters",
    
    # DevOps/Infrastructure
    "Describe Kubernetes architecture and its main components",
    "What are the benefits of containerization? How does Docker help?",
    "Explain the difference between VMs and containers",
    "What is Infrastructure as Code and why is it important?",
    "Design a CI/CD pipeline for a microservices application",
    
    # Security
    "What are the top 5 web application security vulnerabilities (OWASP Top 10)?",
    "Explain authentication vs authorization",
    "How does encryption protect data? Symmetric vs asymmetric",
    "What is a zero-trust security model?",
    "Describe CORS and its security implications",
    
    # Performance
    "What techniques can optimize database query performance?",
    "How do you identify and resolve performance bottlenecks?",
    "Explain caching strategies: TTL, LRU, and distributed caching",
    "What is the n+1 query problem and how do you solve it?",
    "How do you optimize frontend performance?",
    
    # Testing
    "What are the different types of testing? Unit, integration, E2E?",
    "How do you write effective unit tests?",
    "What is test-driven development (TDD) and its benefits?",
    "Explain mocking and stubbing in tests",
    "How do you measure code coverage?",
    
    # Challenging scenarios
    "You have a service that takes 5 seconds to respond but users expect sub-100ms latency. How do you solve this?",
    "A database query used to run in 100ms but now takes 5 seconds. How do you debug this?",
    "Your application crashes under load but works fine in staging. What could be wrong?",
    "A feature works locally but fails in production. How would you troubleshoot?",
    "Users report data inconsistency after a deployment. How do you investigate?",
    
    # Best practices/design
    "What makes a good API design? List key characteristics",
    "Explain the SOLID principles with examples",
    "What is the difference between OOP and functional programming?",
    "Describe common design patterns: Singleton, Factory, Observer",
    "What is DRY principle and why is it important?",
    
    # Emerging tech
    "What is blockchain and how does it work?",
    "Explain serverless computing. When is it a good choice?",
    "What is edge computing and its use cases?",
    "Describe the role of AI/ML in modern software development",
    "What is quantum computing and how might it affect current encryption?",
    
    # Additional diverse prompts
    "How would you build a real-time chat application?",
    "Design a URL shortener like bit.ly",
    "Create a recommendation system for Netflix-like streaming",
    "How do you build a search engine?",
    "Design a rate limiting system for an API",
    
    "What are the differences between HTTP/1.1, HTTP/2, and HTTP/3?",
    "Explain the OSI model and its 7 layers",
    "How does DNS work?",
    "What is a CDN and how does it improve performance?",
    "Describe the SSL/TLS handshake process",
    
    "Generate 5 creative startup ideas in the AI space",
    "What skills should a full-stack developer have?",
    "How do you decide between building vs buying software?",
    "Explain technical debt and how to manage it",
    "What makes a good code review?",
    
    "How do you scale a web application from 1K to 1M users?",
    "Design a system for real-time stock price updates",
    "Create an architecture for a ride-sharing app",
    "Build a notification system that handles billions of events",
    "Design a video streaming platform architecture",
    
    # Additional 15 prompts to reach 100
    "Explain the difference between REST and GraphQL APIs",
    "What is serverless architecture? What are its advantages and disadvantages?",
    "How do you implement rate limiting in a distributed system?",
    "Describe the ACID properties in database transactions",
    "What is a reverse proxy and how does it differ from a forward proxy?",
    "Explain eventual consistency in distributed systems",
    "How do you debug a memory leak in a Node.js application?",
    "What is horizontal scaling vs vertical scaling?",
    "Describe the Twelve-Factor App methodology",
    "How do you implement feature flags in a production system?",
    "Explain the difference between stateless and stateful architectures",
    "What is a circuit breaker pattern and when should you use it?",
    "How do you handle timezone issues in a global application?",
    "Describe the builder pattern in software design",
    "What are the challenges of maintaining legacy code?",
    "Design a distributed cache invalidation strategy",
    "Explain the saga pattern for distributed transactions",
    "How do you implement graceful degradation in services?",
    "What is the bulkhead pattern and when to use it?",
    "Describe the ambassador pattern in microservices",
    "How do you handle idempotency in APIs?",
    "Explain the strangler Fig pattern for legacy system migration",
    "What are the principles of reactive programming?",
    "Design a webhook retry mechanism with exponential backoff",
    "How do you implement multi-tenancy in SaaS applications?",
    "Explain the observer pattern and its real-world applications",
    "What is the CQRS pattern and when would you use it?",
    "How do you design for compliance and data residency requirements?",
]

# Ensure we have at least 100 prompts
assert len(DIVERSE_PROMPTS) >= 100, f"Need at least 100 prompts, got {len(DIVERSE_PROMPTS)}"


def send_prompt(prompt: str, provider: Optional[str] = None) -> dict:
    """Send a prompt to the gateway and return the response."""
    payload = {
        "prompt": prompt,
    }
    if provider:
        payload["provider"] = provider
    
    try:
        response = requests.post(
            GENERATE_ENDPOINT,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Failed to call gateway: {e}")


def run_load_test(num_prompts: int = 100):
    """Run the load test with num_prompts diverse prompts."""
    print(f"🚀 Starting load test: {num_prompts} prompts")
    print("=" * 80)
    
    # Select prompts (cycle if needed)
    prompts_to_send = []
    for i in range(num_prompts):
        prompts_to_send.append(DIVERSE_PROMPTS[i % len(DIVERSE_PROMPTS)])
    
    # Tracking metrics
    results = []
    provider_counts = defaultdict(int)
    quality_scores = []
    latencies = []
    costs = []
    errors = 0
    
    start_time = time.time()
    
    for idx, prompt in enumerate(prompts_to_send, 1):
        try:
            print(f"[{idx:3d}/{num_prompts}] Sending: {prompt[:60]}...", end=" ", flush=True)
            
            response = send_prompt(prompt)
            
            # Extract metrics
            provider = response.get("provider", "unknown")
            latency = response.get("latency_ms", 0)
            cost = response.get("cost", {}).get("actual_usd", 0)
            quality = response.get("quality", {})
            quality_score = quality.get("score", 0)
            quality_label = quality.get("label", "unknown")
            
            provider_counts[provider] += 1
            latencies.append(latency)
            costs.append(cost)
            quality_scores.append(quality_score)
            
            results.append({
                "prompt": prompt[:60],
                "provider": provider,
                "latency_ms": latency,
                "cost_usd": cost,
                "quality_score": quality_score,
                "quality_label": quality_label,
            })
            
            print(f"✓ {latency}ms | ${cost:.6f} | Q:{quality_label} ({quality_score})")
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)[:50]}")
            errors += 1
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("📊 LOAD TEST SUMMARY")
    print("=" * 80)
    
    successful = len(results)
    total_cost = sum(costs)
    
    print(f"✓ Successful:     {successful}/{num_prompts}")
    print(f"✗ Failed:         {errors}/{num_prompts}")
    print(f"⏱️  Total time:      {total_time:.2f}s")
    print(f"⚡ Avg throughput: {successful / total_time:.2f} requests/sec")
    
    print(f"\n💰 COST METRICS")
    print(f"  Total cost:      ${total_cost:.6f}")
    print(f"  Avg cost/req:    ${statistics.mean(costs):.6f}" if costs else "  N/A")
    print(f"  Max cost:        ${max(costs):.6f}" if costs else "  N/A")
    print(f"  Min cost:        ${min(costs):.6f}" if costs else "  N/A")
    
    print(f"\n⏰ LATENCY METRICS (ms)")
    if latencies:
        print(f"  Avg latency:     {statistics.mean(latencies):.2f}")
        print(f"  Median latency:  {statistics.median(latencies):.2f}")
        print(f"  Max latency:     {max(latencies):.2f}")
        print(f"  Min latency:     {min(latencies):.2f}")
        print(f"  Stdev:           {statistics.stdev(latencies) if len(latencies) > 1 else 0:.2f}")
    
    print(f"\n⭐ QUALITY METRICS")
    if quality_scores:
        print(f"  Avg quality:     {statistics.mean(quality_scores):.4f}")
        print(f"  Median quality:  {statistics.median(quality_scores):.4f}")
        print(f"  Max quality:     {max(quality_scores):.4f}")
        print(f"  Min quality:     {min(quality_scores):.4f}")
    
    quality_distribution = defaultdict(int)
    for result in results:
        quality_distribution[result["quality_label"]] += 1
    print(f"  Distribution:")
    for label in ["excellent", "good", "fair", "poor"]:
        count = quality_distribution.get(label, 0)
        percent = (count / successful * 100) if successful else 0
        print(f"    {label:10s}: {count:3d} ({percent:5.1f}%)")
    
    print(f"\n🏪 PROVIDER DISTRIBUTION")
    for provider, count in sorted(provider_counts.items()):
        percent = (count / successful * 100) if successful else 0
        print(f"  {provider:15s}: {count:3d} ({percent:5.1f}%)")
    
    print("\n" + "=" * 80)
    print(f"Load test complete. Data persisted to Postgres for analytics queries.")
    print("=" * 80)
    
    return {
        "total_requests": num_prompts,
        "successful": successful,
        "errors": errors,
        "total_time": total_time,
        "total_cost": total_cost,
        "avg_latency": statistics.mean(latencies) if latencies else 0,
        "avg_quality": statistics.mean(quality_scores) if quality_scores else 0,
        "results": results,
    }


if __name__ == "__main__":
    run_load_test(100)
