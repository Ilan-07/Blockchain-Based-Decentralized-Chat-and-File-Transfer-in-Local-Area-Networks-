# 🧩 Blockchain-Based LAN Chat — Secured Interactive Version

### 🚀 Overview
This project implements a **fully interactive, secure, and decentralized LAN chat application** built on **blockchain principles**.  
Each message or file transfer is recorded as a **block**, ensuring **tamper-proof communication** and **transparent synchronization** among all peers in the local network.

### 🛡️ Key Features
- **Blockchain Backbone:** Every message is stored as a verifiable block.  
- **Peer Synchronization:** Automatic request/response protocol ensures full chain consistency across all connected peers.  
- **Tamper Detection:** Detects and broadcasts file or block modifications with live alerts.  
- **Real-Time Metrics:** Graphical analysis of latency, throughput, and packet loss.  
- **Blockchain Visualizer:** Interactive view of the chain, with block-level details.  
- **Cross-Platform Notifications:** Desktop alerts for incoming messages or tamper events.  
- **Persistent Peer Colors:** Each node is color-coded for easy identification.  
- **Enhanced Security:**  
  - Thread-safe blockchain access  
  - Input sanitization  
  - Path traversal prevention  
  - Rate-limiting and message size checks  
  - Bounded memory usage  

---

### 🧰 Requirements
Make sure you have the following dependencies installed:

```bash
pip install customtkinter matplotlib networkx plyer win10toast
```

> 💡 Note: Some dependencies (like `win10toast`) are optional and used only for Windows notifications.

---

### ⚙️ How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/bccc.git
   cd bccc
   ```

2. Run the main script:
   ```bash
   python3 bccc.py
   ```

3. Launch the application on **multiple computers** connected to the same **LAN network**.  
   Each instance will auto-detect peers, sync blockchain data, and display a visual dashboard.

---

### 🧭 Interface Overview
- **Chat Panel:** Send and receive messages across all peers.  
- **Visualizer Panel:** Explore the blockchain, inspect block data, and monitor peer performance.  
- **Graph Panel:** Live updates on network latency and throughput.  
- **Tamper Alerts:** Real-time warning and quarantine mechanism for corrupted data.

---

### 🧱 System Architecture
```text
+-----------------------------+
|        User Interface       |
| (CustomTkinter + Matplotlib)|
+-------------+---------------+
              |
              v
+-----------------------------+
|       Blockchain Engine     |
|  (Block creation, hashing,  |
|  tamper verification, sync) |
+-------------+---------------+
              |
              v
+-----------------------------+
|      Networking Layer       |
|   (Sockets, threading, RTT) |
+-----------------------------+
```

---

### 🧩 Security Enhancements
- Enforced **thread locks** for safe blockchain modification  
- **Timeouts** and **rate limiting** to prevent flooding  
- Sanitized **user input** and **path handling**  
- **Context-managed resources** to prevent leaks  
- Verified **peer requests** to avoid spoofing  

---

### 🧪 Future Improvements
- Encrypted peer-to-peer messaging  
- Global key exchange mechanism  
- Web dashboard for blockchain inspection  
- Support for audio/file transfer blocks  

---

### 👨‍💻 Author
**Ilangkumaran Yogamani**  
📧 *ilangkumaran.2024@vitstudent.ac.in*
**Ranen Abner**  
📧 *ranen.abner2024@vitstudent.ac.in*

---

### 📜 License
This project is licensed under the **MIT License** — free to use and modify with proper credit.

---

### 🧠 Acknowledgements
- **Python 3.10+**
- **CustomTkinter** for modern UI  
- **Matplotlib + NetworkX** for visual analytics  
- **Socket Programming** for LAN-based communication  
- **Blockchain architecture** for immutability and transparency

---

> 🧱 *“Secure communication doesn’t just connect nodes — it connects trust.”*
