# RAG-Enhanced Large Language Model for Policy Generation in IoT Intrusion Detection Systems

## Introduction

In this research we try to find how LLMs perform to improve the IoT security and access control.

## Getting Started

1. Clone the repository.
2. Install required Python libraries using `pip install -r requirements.txt`.
3. Create `.env` file and add API keys.

    ```bash
    HUGGINGFACEHUB_API_TOKEN=
    OPENAI_API_KEY=
    GOOGLE_API_KEY=
    ANTHROPIC_API_KEY=
    ```

4. Create the `data` directory and sub directories as follows.

    ```
    iot-llm
      └-data
        |-cic-iot
        |-wustl-iiot
        |-ton-iot
        |-bot-iot
        └-unsw-nb15
    ```

5. Place downloaded datasets in the relevant directories.

6. Run Python notebooks in order for each dataset.

## Datasets

| Name | Paper(s) | Year |
| - | - | - |
| CICIoT2023* | CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment | 2023 |
| Edge-IIoTSet | Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning | 2022 |
| WUSTL-IIoT* | WUSTL-IIOT-2021 Dataset for IIoT Cybersecurity Research | 2021 |
| IoT-23 | IoT-23: A labeled dataset with malicious and benign IoT network traffic | 2020 |
| TON_IoT* | TON_IoT telemetry dataset: a new generation dataset of IoT and IIoT for data-driven Intrusion Detection Systems | 2020 |
| Bot-IoT* | Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-iot dataset | 2019 |
| N-BaIoT | N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders | 2018 |
| UNSW-NB15* | UNSW-NB15: A Comprehensive Data set for Network Intrusion Detection systems | 2015 |

\* Used for our experiment.

## Large Language Models

| Name | Provider |
|-|-|
| gpt-4o (gpt-4o-2024-05-13) | OpenAI |
| gemini-1.5-pro | Google |

