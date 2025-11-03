# 🧩 Blockchain-Based LAN Chat — Secured Interactive Version

### 🚀 Overview
This project implements a **fully interactive, secure, and decentralized LAN chat application** built on **blockchain principles**.  
Each message or file transfer is recorded as a **block**, ensuring **tamper-proof communication** and **transparent synchronization** among all peers in the local network.

---

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

## ⚙️ System Requirements

- **Python:** Version 3.10 or later
- **Operating System:** Windows / macOS / Linux
- **Network:** All peers connected to the same LAN (via WiFi or Ethernet)

---

## 🐍 Install Python (Step-by-step)

Follow the instructions for your operating system to install Python and prepare the environment.

### macOS (recommended: Homebrew)
1. **Install Homebrew** (if you don't have it):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. **Install Python 3.10+ with Homebrew**:
   ```bash
   brew update
   brew install python@3.10
   ```
3. **Ensure `python3` and `pip3` refer to the installed Python**:
   ```bash
   python3 --version
   pip3 --version
   ```
4. (Optional) **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

### Windows
1. **Download the Python installer** from the official site: https://www.python.org/downloads/windows/ (choose 3.10+).  
2. **Run the installer** and **check “Add Python to PATH”** on the first installation screen.  
3. After installation, verify in PowerShell / Command Prompt:
   ```powershell
   python --version
   pip --version
   ```
4. (Optional) **Create and activate a virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

### Linux (Ubuntu / Debian)
1. **Update package list**:
   ```bash
   sudo apt update
   ```
2. **Install Python 3.10+ and pip**:
   ```bash
   sudo apt install -y python3 python3-venv python3-pip
   ```
3. **Verify installation**:
   ```bash
   python3 --version
   pip3 --version
   ```
4. (Optional) **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

> **Note:** If your distribution uses `python` for Python 2, use `python3` and `pip3` commands instead.

---

## 🧩 Dependencies

Make sure these dependencies are installed (automatically handled via `requirements.txt`):

```
customtkinter==5.2.2
plyer==2.1.0
matplotlib==3.8.0
networkx==3.2.1
numpy==1.26.0
pandas==2.1.1
uuid==1.30
```

To install them manually:
```bash
pip install -r requirements.txt
```

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


## 📖 Step-by-Step User Manual

### 🪜 1. Setup and Launch
1. Ensure **Python 3.8+** is installed (`python --version`).
2. Install required libraries using:
   ```bash
   pip install customtkinter matplotlib networkx plyer win10toast
   ```
3. Place the `bccc.py` file and any supporting files in a single folder.
4. Open a terminal or command prompt in that folder.
5. Start the program:
   ```bash
   python3 bccc.py
   ```
6. Repeat the same steps on **all other systems** connected to the same LAN.

---

### 💬 2. Sending Messages
- Type your message in the **Chat Input Box** at the bottom of the window.  
- Press **Enter** or click **Send**.  
- The message is added as a **new block** and broadcast to all peers.  
- Other peers will see your message instantly — synchronized through the blockchain.

---

### 📁 3. File Transfer
- Click the **Attach / Upload** button (if available) to select a file.
- The selected file will be converted into a block and shared with all peers.
- The blockchain records the transfer to ensure integrity and traceability.
- All peers can view or download the file from the shared ledger.

---

### 🔍 4. Viewing the Blockchain
- Open the **Blockchain Visualizer** tab in the interface.
- Each block represents a message or file event.
- Click on a block to view:
  - Sender information  
  - Timestamp  
  - Hash and previous hash values  
  - Data integrity status  

---

### ⚙️ 5. Monitoring Performance
- The **Performance Graph** shows:
  - **Latency:** round-trip time between peers.
  - **Throughput:** rate of data transfer.
  - **Packet Loss:** any message drops or network delays.
- These metrics update in real-time while chatting or transferring files.

---

### 🚨 6. Tamper Detection
- If a peer modifies or deletes a block:
  - The system immediately detects the mismatch.
  - All peers receive a **Tamper Alert** popup and notification.
  - The affected block is quarantined or marked invalid.
- You can view alerts in the blockchain visualizer.

---

### 🎨 7. Color-Coded Peers
- Each peer (device) is assigned a **unique color** for clarity in the UI.
- The color mapping persists between sessions using a local JSON file (`peer_colors.json`).

---

### 🧱 8. Quitting Safely
- To exit the program:
  - Close the window, or  
  - Use **Ctrl + C** in the terminal.
- The program automatically closes sockets and saves blockchain state.

---

### 🧠 9. Troubleshooting
| Issue | Possible Cause | Solution |
|-------|----------------|-----------|
| Peers not visible | Devices not on same network | Check LAN/Wi-Fi connection |
| No messages appearing | Firewall blocking ports | Allow Python in firewall |
| Visualizer not updating | Matplotlib animation paused | Restart the app |
| Tamper alert triggered unexpectedly | Manual modification or sync delay | Wait for auto-sync or restart all peers |

---

### 🧩 10. Best Practices
- Run the app on a **stable LAN** with good connectivity.
- Avoid renaming or deleting blockchain files while running.
- Keep all peers on the **same version** of the application.
- For testing, run multiple terminal windows on a single machine with different ports.

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

## 🧪 Testing Scenarios

### 🔧 0. **Node Configuration Setup (Before Testing)**

- When you launch the application, you’ll see the “Configure Your Node” screen (as shown above).
- Here’s what each field means:
- Node Name: A custom name for your node (e.g., Alice’s Node, Node_1, etc.).
- Your Port: The local port number for your node (default is 5001).
- Connect to Peer (Optional): Enter another peer’s IP and port (e.g., 192.168.1.100:5001) if you want to connect to an existing node.
- Leave it blank if this is the first node on the network.
- Once all fields are filled, click 🚀 Start Node to initialize your blockchain node.

### 1. **Blockchain Synchronization Test**
- Run the app on **two or more computers** connected to the **same LAN**.
- Send messages or files between peers.
- Verify that each blockchain visualizer shows **identical blockchains** (same number of blocks, hashes, and timestamps).

### 2. **Tamper Detection Test**
- Manually modify a stored file or blockchain record.
- Observe the **tamper alert** in the blockchain visualizer.
- The affected block will be highlighted and the alert broadcast to all peers.

### 3. **Performance Analysis Test**
- Send multiple large files or frequent messages.
- Watch the **real-time graph** for throughput and latency performance.
- Compare different runs to analyze network behavior.

### 4. **File Transfer Verification**
- Upload a file using the chat interface.
- Confirm that all connected peers receive the file.
- Verify that a new blockchain block is created containing the file’s metadata.

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

## 🏁 Conclusion

The project successfully demonstrates how **blockchain principles** can be applied to create a **secure, decentralized communication system** within a LAN.
It ensures **data integrity**, **peer-to-peer transparency**, and **real-time synchronization** — paving the way for more resilient local communication networks.


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
