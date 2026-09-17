# Orbit Platform Runbook

## Overview
Orbit is the internal platform that serves machine learning models for the Northwind Robotics fleet. It runs a FastAPI gateway in front of a pool of inference workers. Each request receives a request ID at the gateway, and that ID is attached to every log line the request produces.

## Clusters
Orbit runs on two Kubernetes clusters. The production cluster, called orbit-prod, has 12 GPU worker nodes, each with four NVIDIA L4 GPUs. The staging cluster, orbit-stage, has 3 GPU worker nodes with one L4 GPU each. All deployments reach staging first and must run there for at least 24 hours before promotion to production.

## Service level objectives
The p95 latency target for the gateway is 800 milliseconds for embedding requests and 4 seconds for text generation requests. The monthly availability target is 99.9 percent. If the error rate stays above 2 percent for 10 minutes, the on-call engineer is paged automatically.

## Logging and metrics
Logs are shipped to a central log store and kept for 30 days. Metrics include request count, error rate, queue time, time to first token, and GPU memory usage. Dashboards for Orbit live in the observability workspace under the folder named Orbit Serving.

## Deployments
Model deployments use a canary strategy. A new model version first receives 5 percent of traffic for one hour. If error rate and latency stay within the service level objectives, traffic moves to 50 percent and then to 100 percent. Rollbacks are performed with the command orbitctl rollback followed by the service name.

## On-call
The on-call rotation changes every Monday at 09:00. The primary on-call engineer must acknowledge a page within 10 minutes. If the primary does not respond, the page escalates to the secondary engineer after 15 minutes. Incidents rated severity 1 require a written postmortem within five working days.

## Capacity and scaling
Inference workers scale horizontally based on queue depth. When the average queue time exceeds 500 milliseconds for five minutes, the autoscaler adds workers, up to a maximum of 40 workers in production. GPU memory usage above 90 percent on any node triggers a warning alert.
