
# Reply Mirror – AI Agent Challenge 2026

**Date:** 12th March 2026

## Abstract

In the year 2087, the digital metropolis of **Reply Mirror** thrives as a model of futuristic transparency, where every citizen’s data is part of a vast, publicly accessible information lattice. Financial institutions like **MirrorPay** manage not only economic transactions but hold the deepest reservoirs of demographic and behavioral data. Amidst this urban data symphony, only one entity stands between order and chaos: **The Eye**—a superintelligent agent tasked exclusively with detecting and neutralizing sophisticated financial fraud.

MirrorPay’s battle is far from routine. Its adversaries—**Mirror Hackers**—are relentlessly strategic, adept at morphing their attack patterns, and unafraid to exploit every gap in the dynamic regulatory and geographic fabric of the city’s commerce. Your mission: build an AI agent system that adapts, learns, and outsmarts these attackers, keeping the fabric of Reply Mirror’s economy intact.

---

## Section 1: Problem Statement

- **Year 2087**: Reply Mirror thrives on transparency with publicly accessible personal data.
- **MirrorPay** manages economic transactions and holds vast demographic data.
- **Role of "The Eye"**: detect and neutralize financial fraud.
- **Mirror Hackers**: smart, adaptive, relentlessly strategic adversaries.
    - Their tactics:
        - targeting new merchants and transaction categories
        - shifting temporal habits (e.g. daytime to late-night activity)
        - operating across changing geographic regions and jurisdictions
        - varying transaction amounts and frequency
        - creating new, deceptive behavioural sequences
- **Static models will fail**; only dynamic, continuously learning and strategically adaptive solutions can succeed.
- **Challenge Flow**: unfolds in five levels, each with a training dataset and an evaluation dataset.
- Only first submission per level is accepted and considered final.
- Teams must design a system of cooperative intelligent agents to identify anomalous activities.
- **Score** depends on adaptability and accuracy across all five levels.
- **Challenge Goal**: design an agent-based system capable of:
    - detecting fraudulent behaviour that evolves over time
    - anticipating new attack patterns using memory of past interactions
    - responding in real time to sudden changes without degrading performance
    - keeping the false positive rate low
- **Asymmetric cost model:**
    - **False positive** (blocking legitimate transaction): economic and reputational losses.
    - **False negative** (allowing fraudulent transaction): significant financial damage.
- **Final Score**: accuracy + temporal stability + adaptability.

---

## Section 2: Input Format

- **Transactions.csv** with T records including:
    - Transaction ID
    - Sender ID
    - Recipient ID
    - Transaction Type: bank transfer, in-person payment, e-commerce, direct debit, withdrawal
    - Amount
    - Location (only for in-person payments)
    - Payment Method: debit card, mobile device, smartwatch, GooglePay, PayPal
    - Sender IBAN (only for bank transfers)
    - Recipient IBAN (only for bank transfers)
    - Balance
    - Timestamp
- **Locations**: geo-referenced GPS data with BioTag, Datetime, Lat, Lng
- **Users**: resume of citizen's personal data
- **Conversations**: SMS threads with User ID and complete textual thread
- **Messages**: e-mail interactions with mail (complete textual thread)

---

## Section 3: Output Format

- **ASCII text file**
- Each line: one suspected fraudulent Transaction ID (`t`)
- Output is invalid if:
    - no transactions are reported
    - all transactions are reported
    - less than 15% of the fraudulent transactions are correctly identified

---

## Section 4: Scoring Rules

- **Composite scoring model** with two key dimensions: economic impact (primary) and accuracy
- **Goal:** economically sustainable, operationally efficient AI multi-agent system
- **Accuracy:** balanced trade-off between fraud detection and avoiding unjustified customer impact
- **Additional Metrics:** cost, speed, efficiency — reward optimized agent architecture with real-time fraud detection and low operational expense; measure scalability, latency, and infrastructure usage

---

## Section 5: Example

**Input Example:**
```plaintext
4a92ab00-8a27-4623-ab1d-56ac85fcd6b0,SCHV-SVRA-7BC-COR-0,,e-commerce,56.63,,mobile device,IT16Y9430002300167070752952,IT35O1753705526805948017123071,603.37,,2025-11-17T00:35:29.446363
8830a720-ff34-4dce-a578-e5b8006b2976,LRNT-MTTH-7BF-PAR-1,,prelievo,150,Turin,debit card,FR46C7104822278076244862444,IT39S3051166323954859019873188,142.58,,2025-11-17T14:33:28.068080
1c6db202-22d8-443f-86e7-fb1a8df05e84,TRBU-MRTT-7C4-MUL-0,BTSWF98176,e-commerce,170.33,SwiftCart Marketplace,debit card,DE20X3656132271467727362296,IT14Q8802310964869133978727249,54000.24,,2027-01-04T00:00:00
```

**Output Example:**
```plaintext
4a92ab00-8a27-4623-ab1d-56ac85fcd6b0
8830a720-ff34-4dce-a578-e5b8006b2976
```
