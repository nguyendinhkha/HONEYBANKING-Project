import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import webbrowser
from flask import Flask, request, render_template, session, redirect, url_for, jsonify, flash
from flask_sqlalchemy import SQLAlchemy    # "SQLAlchemy là một bộ công cụ SQL và ORM cho Python, cung cấp một cách để quản lý các hoạt động CSDL hiệu quả hơn.
import re
import logging
import random
import string
from datetime import datetime
import io
import base64
import threading  # Import threading
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from apscheduler.schedulers.background import BackgroundScheduler

import numpy as np 
from dqn_agent import DQNAgent  # Import the DQNAgent

import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from threading import Lock
# Global lock for periodic evaluation
evaluation_lock = Lock()

app = Flask(__name__)

# Setup logging
logging.basicConfig(filename='honeypot.log', level=logging.INFO)

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///honeyBANKING.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = ''.join(random.choices(string.ascii_letters + string.digits, k=16))  # Generate a random secret key

db = SQLAlchemy(app)  # khởi tạo đối tượng SQLAlchemy với ứng dụng Flask, cho phép các chức năng ORM

request_counts = {}
batch_size = 32  # Define the batch size for PER     

# Initialize DQN Agent
state_size = 7  #  state size
action_size = 2  # action size: 0 = no action, 1 = alert
agent = DQNAgent(state_size, action_size)

# Scheduler setup
scheduler = BackgroundScheduler()

# Function to log and plot metrics periodically
def periodic_evaluation():
    with app.app_context():
        log_evaluation_metrics()
        agent.plot_metrics()
        if agent.memory_size() >= batch_size:
            agent.replay(batch_size)
            log_evaluation_metrics()
            agent.plot_metrics()

scheduler.add_job(periodic_evaluation, 'interval', minutes=5)
scheduler.start()

def log_evaluation_metrics():
    with app.app_context():
        metrics = agent.calculate_all_metrics()
        for attack_type, (accuracy, precision, recall, f1) in metrics.items():
            logging.info(f"Evaluation Metrics for {attack_type} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
            print(f"Evaluation Metrics for {attack_type} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
         
# Define the database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    password = db.Column(db.String(50), nullable=False)
    balance = db.Column(db.String(20), nullable=False)
    full_name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    stk = db.Column(db.String(20))

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipient_stk = db.Column(db.String(20), nullable=False)
    recipient_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False,  default=lambda: datetime.now(timezone.utc))
    transaction_type = db.Column(db.String(10), nullable=False)  # 'incoming' or 'outgoing'

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    endpoint = db.Column(db.String(50), nullable=False)
    data = db.Column(db.String(200), nullable=False)
    response = db.Column(db.String(200), nullable=False)


# Initialize the database
def initialize_database():
    with app.app_context():
        db.create_all()
        if User.query.count() == 0:
            # Add mock users
            users = [    
                User(username='0968257909', password='Vendetta235149@@', balance='50,171,255', full_name='Nguyễn Đình Kha', email='dinhkhanguyen2002@gmail.com', stk='8014286847037'),
                User(username='0919909456', password='Th121002anhTR@@', balance='77,552,559', full_name='Tạ Trúc Thanh', email='tathitructhanh051177@gmail.com', stk='9720657882244'),
                User(username='0565988116', password='Catanddog5178tree', balance='64,469,888', full_name='Alice Parker', email='aliceparker@gmail.com', stk='5825623773876'),
                User(username='0918758234', password='Playwithme25178', balance='25,265,592', full_name='Nguyễn Đình Thiện', email='nguyendinhthien@gmail.com', stk='4398963858278'),
                User(username='0022400525', password='%M&1vl\\3^fyjG%}:', balance='22,038,038', full_name='Đặng Ngọc Trang', email='dangngoctrang@gmail.com', stk='5087669646481'),
                User(username='0132384327', password='Mj[<"p>NstK00Q<:', balance='94,625,048', full_name='Nguyễn Phúc Thu', email='nguyenphucthu@gmail.com', stk='1292758972950'),
                User(username='0695273572', password='jXuE=}uA5-\\\\kA~&', balance='65,340,779', full_name='Đỗ Nhật Thu', email='donhatthu@gmail.com', stk='6034714286676'),
                User(username='0660027655', password="K?,z'hz^77_MO3TM", balance='50,734,830', full_name='Võ Thanh Bình', email='vothanhbinh@gmail.com', stk='7958477675323'),
                User(username='0087350653', password='N}Qr=c)K+,m7z^7Y', balance='56,843,791', full_name='Trần Thanh Linh', email='tranthanhlinh@gmail.com', stk='3732428662904'),
                User(username='0805945209', password='$!fd6Th7%K`B3ayy', balance='38,211,504', full_name='Bùi Hữu Sơn', email='buihuuson@gmail.com', stk='4142835995067'),
                User(username='0685134357', password='D+/r2`@{Fnv2\\w"\'', balance='13,257,776', full_name='Phan Ngọc Châu', email='phanngocchau@gmail.com', stk='1179461923974'),
                User(username='0059245360', password='pqPs4JWAxe|=-9+S', balance='24,593,718', full_name='Đỗ Quốc Quân', email='doquocquan@gmail.com', stk='0962061885139'),
                User(username='0313603204', password="%LP\\]5Gp`u`'}9`l", balance='59,900,237', full_name='Vương Quang Lan', email='vuongquanglan@gmail.com', stk='8248430984196'),
                User(username='0615628032', password='|\\oZXTncRWlDU#_T', balance='22,809,796', full_name='Đỗ Bích Quân', email='dobichquan@gmail.com', stk='3562607941912'),
                User(username='0065605827', password='5qNySC4\\\\@}`9iZ[', balance='78,893,147', full_name='Trịnh Thị Dũng', email='trinhthidung@gmail.com', stk='2368728332825'),
                User(username='0454351588', password='/6M\\eNRypF<cdZmH', balance='53,657,575', full_name='Võ Thanh Quân', email='vothanhquan@gmail.com', stk='3586486717955'),
                User(username='0023562263', password='*%J|o#qPl6-zT[G!', balance='14,020,218', full_name='Huỳnh Minh Quân', email='huynhminhquan@gmail.com', stk='3142859707476'),
                User(username='0466282729', password='))8n.{f05O-aZD!Y', balance='83,109,628', full_name='Trần Hồng Quân', email='tranhongquan@gmail.com', stk='4399125331404'),
                User(username='0360559859', password='c~X]$(=,^&OedzYW', balance='70,589,690', full_name='Võ Thị Trang', email='vothitrang@gmail.com', stk='5147195347051'),
                User(username='0102539948', password='5ISiKb,`W55\\gRq#', balance='19,312,749', full_name='Đặng Kim Lệ', email='dangkimple@gmail.com', stk='1629786779267'),
                                  
            
            ]

            db.session.bulk_save_objects(users)
            db.session.commit()

initialize_database()

# DeepDIg
def generate_fake_data():
    fake_users = [
        { 'username': '0002536765', 'password': 'Kf;Ho%4U`(P3"\'nW', 'balance': '86,090,711', 'full_name': 'Đặng Nhật Linh', 'email': 'dangnhatlinh@gmail.com', 'stk': '8295290731071' },
        { 'username': '0893903373', 'password': "WoW;-!5(eCM._']6",  'balance': '23,782,341', 'full_name': 'Lê Kim Bình',  'email': 'lekimbinh@gmail.com', 'stk': '8355167737428' },
        { 'username': '0940612475', 'password': 'zwqE1$P#UhH":){u', 'balance': '25,176,998', 'full_name': 'Bùi Minh Thảo', 'email': 'buiminhthao@gmail.com', 'stk': '2580154515702' },
        { 'username': '0899592621', 'password': '\\m~:>Rsq)-Z-#.hi', 'balance': '43,439,043', 'full_name': 'Phạm Thiên Sơn', 'email': 'phamthienson@gmail.com', 'stk': '5538381504998' },
        { 'username': '0284443985', 'password': 'IZi^jdq<P9wg@`ba', 'balance': '90,187,715', 'full_name': 'Võ Nhật Thảo', 'email': 'vonhatthao@gmail.com', 'stk': '0811456638902' },
        { 'username': '0783454998', 'password': '_,JX0(8jt[:y^`Em', 'balance': '65,981,451', 'full_name': 'Đặng Quỳnh Mai', 'email': 'dangquynhmai@gmail.com', 'stk': '2597149199685' },
        { 'username': '0063734431', 'password': 'qko0yzY"\'VrZk/JV', 'balance': '42,736,209', 'full_name': 'Hoàng Kim Khánh', 'email': 'hoangkimkhanh@gmail.com', 'stk': '6212543287568' },
        { 'username': '0788054067', 'password': 'jfNq(x66D81a&[R>', 'balance': '18,562,472', 'full_name': 'Vũ Minh Sơn', 'email': 'vuminhson@gmail.com', 'stk': '2326224143271' },
        { 'username': '0536875308', 'password': 'g[x"|7L<t!VY_*9u', 'balance': '89,277,280', 'full_name': 'Đỗ Thanh Trang', 'email': 'dotranganh@gmail.com', 'stk': '9685894501362' },
        { 'username': '0958344628', 'password': 'qsKFJ5=v/=2Q45H\\', 'balance': '33,171,891', 'full_name': 'Lê Quốc Lan', 'email': 'lequoclan@gmail.com', 'stk': '6492335343917' },
        { 'username': '0161101028', 'password': '6[PR9F#u%?P9!2X6', 'balance': '71,676,154', 'full_name': 'Lý Quỳnh Trang', 'email': 'lyquynhtrang@gmail.com', 'stk': '9457871408285' },
        { 'username': '0595953655', 'password': '?I|26FJ"~fwC>bv`', 'balance': '28,088,203', 'full_name': 'Võ Hồng Phương', 'email': 'vohongphuong@gmail.com', 'stk': '7918691399828' },
        { 'username': '0305645047', 'password': '8[tU:jJ#jVLDTMoe', 'balance': '51,549,055', 'full_name': 'Tô Thiên Khánh', 'email': 'tothienkhanh@gmail.com', 'stk': '1168705087013' },
        { 'username': '0710857944', 'password': 'N/q$,{29^7El+`My', 'balance': '69,738,930', 'full_name': 'Huỳnh Nhật Châu', 'email': 'huynhnhatchau@gmail.com', 'stk': '6239073031186' },
        { 'username': '0958073149', 'password': "'CkkscN8cMdZM(67", 'balance': '16,426,312', 'full_name': 'Vũ Thanh Quân', 'email': 'vuthanhquan@gmail.com', 'stk': '2421137454614' },
        { 'username': '0002906824', 'password': '~]#h-S-CcXar3g!)', 'balance': '25,541,279', 'full_name': 'Hồ Thiên Huy', 'email': 'hothienhuy@gmail.com', 'stk': '0974205595505' },
        { 'username': '0339379173', 'password': '=B6,S`v%_AL^194H', 'balance': '65,398,723', 'full_name': 'Phan Mai Hà', 'email': 'phanmaiha@gmail.com', 'stk': '9848586539090' },
        { 'username': '0298302296', 'password': '6(+uYZR8k<+y6Rov', 'balance': '52,038,656', 'full_name': 'Tô Hương Châu', 'email': 'tohuongchau@gmail.com', 'stk': '0464539660045' },
        { 'username': '0024851492', 'password': '9.xaspq}j:n6XLyl', 'balance': '95,617,810', 'full_name': 'Ngô Thị Thu', 'email': 'ngothithu@gmail.com', 'stk': '7798238098686' },
        { 'username': '0609016331', 'password': '2g2HwrCkld<>TB(t', 'balance': '97,680,371', 'full_name': 'Ngô Thiên Thảo', 'email': 'ngothienthao@gmail.com', 'stk': '4421095563108' }
    ]
    return fake_users

# Track login attempts
login_attempts = {}

blocklist = set()

# Initialize the XSS attack attempt counters
xss_attack_attempt_counter = 0

# Initialize a counter for SQL Injection attempts
sql_injection_attempt_counter = 0

def detect_sql_injection(input_string):
  
    sql_injection_patterns = [
        # Basic SQL Injection patterns
        r"(%27)|(')|(--)|(%23)|(#)",  # Detects ' or -- or #
        r"\b(SELECT|UPDATE|DELETE|INSERT|DROP|ALTER|CREATE|RENAME|TRUNCATE|REPLACE|MERGE|CALL|EXPLAIN|DESCRIBE)\b",  # SQL keywords
        r"\b(OR|AND)\b\s+\b(=|IS|LIKE)\b",  # Logical operators OR and AND followed by =

        # Error-based SQL Injection patterns
        r"(?i)\b(ERROR|WARNING|ERROR_MESSAGE|PRINT)\b",  # SQL error messages
        r"(?i)\b(SYNTAX|ERROR)\b",  # Common error keywords

        # Time-based SQL Injection patterns
        r"(?i)\b(WAITFOR\s+DELAY|BENCHMARK\s*\(\s*[0-9]+,\s*SHA1\s*\()\b",  # Time delay keywords
        r"(?i)\b(SLEEP\(\s*[0-9]+\s*\))\b",  # Sleep function used in time-based injections

        # Union-based SQL Injection patterns
        r"(?i)\b(UNION\s+SELECT)\b",  # UNION SELECT pattern
        r"(?i)\b(UNION\s+ALL\s+SELECT)\b",  # UNION ALL SELECT pattern
        r"(?i)\b(SELECT\s+NULL)\b"  # SELECT NULL pattern used in UNION-based injections

        # Extended SQL Injection patterns
        r"(%2D%2D)|(--)",  # Detects URL encoded comments
        r"\b(UNION|SELECT|ORDER|BY|GROUP|HAVING|DATABASE|SCHEMA|TABLE|COLUMN|EXEC|EXECUTE|DECLARE|NVARCHAR|SP_|XP_|CAST|CONVERT|SYSOBJECTS|SYSUSERS|SYSCOLUMNS|SYSDATABASES|SYSFILES|SYSINDEXES|SYSKEYS|SYSMEMBERS|SYSMESSAGES|SYSPERMISSIONS|SYSPROCEDURES|SYSSCHEMAS|SYSTYPES|SYSUSAGES|SYSVIEWS)\b",  # Extended SQL keywords
        r"\b(0x[0-9A-Fa-f]+)\b"  # Hexadecimal patterns often used in SQL injections
    ]

    for pattern in sql_injection_patterns:
        if re.search(pattern, input_string, re.IGNORECASE):
            logging.info(f"Phát hiện tấn công SQL Injection: {input_string}")
            global sql_injection_attempt_counter
            sql_injection_attempt_counter += 1
            print(f"SQL Injection attempts detected: ", sql_injection_attempt_counter)
            return True
    return False

@app.before_request
def log_request():
    # Log the incoming request details
    logging.info(f"Request: {request.method} {request.url} {request.data.decode()}")


def detect_xss(input_string):
    xss_patterns = [
        r"<script.*?>.*?</script.*?>",  # Simple script tag detection
        r"on\w+\s*=\s*(['\"]?)[^>]+?\1",  # Inline event handlers like onclick, onmouseover, etc.
        r"javascript:",  # JavaScript URI
        r"<.*?javascript:.*?>",  # Tags with JavaScript URI
        r"document\.cookie",  # Accessing cookies
        r"document\.write",  # Writing to the document
        r"window\.location",  # Redirecting the window
        r"eval\(",  # Using eval
        r"alert\(",  # Using alert
        r"<img\s+src\s*=\s*(['\"]?)\s*javascript\s*:\s*[^>]*>",  # Image tags with JavaScript
        r"<iframe\s+src\s*=\s*(['\"]?)\s*javascript\s*:\s*[^>]*>",  # Iframe tags with JavaScript
        r"<svg.*?>.*?</svg.*?>",  # SVG tags
        r"<svg\s+onload\s*=\s*(['\"]?)[^>]+?\1",  # SVG with onload attribute
        r"expression\s*:\s*\(.*?\)",  # CSS expressions
        r"vbscript:",  # VBScript URI
     
    ]

    for pattern in xss_patterns:
        if re.search(pattern, input_string, re.IGNORECASE):
            logging.info(f"Phát hiện tấn công XSS: {input_string}")
            print(f"Phát hiện tấn công XSS: {input_string}")
            global xss_attack_attempt_counter
            xss_attack_attempt_counter += 1
            print(f"Phát hiện tấn công XSS: ", xss_attack_attempt_counter)
            if xss_attack_attempt_counter >=5000:  # Agent memory_size is 5000
                logging.warning("uộc tấn công XSS đã vượt quá dung lượng bộ nhớ!")
                print(f"uộc tấn công XSS đã vượt quá dung lượng bộ nhớ!")
            return True
    return False


def detect_csrf(referer, origin):
    # Allowing for a list of trusted domains
    trusted_domains = ["localhost", "127.0.0.1", "192.168.136.168", "192.168.136.174", "192.168.136.175"]  # Add trusted domains 

    def extract_domain(url):
        if url:
            return url.split('//')[-1].split('/')[0].split(':')[0]
        return None

    referer_domain = extract_domain(referer)
    origin_domain = extract_domain(origin)

    # Detect CSRF by checking the referer and origin headers against trusted domains
    if referer_domain not in trusted_domains or origin_domain not in trusted_domains:
        logging.info(f"Phát hiện tấn công CSRF: tên miền tham chiếu = {referer_domain}, tên miền gốc = {origin_domain}" )
        return True
    return False

@app.route('/evaluate', methods=['GET'])
def evaluate():
    with app.app_context():
        accuracy, precision, recall, f1 = agent.calculate_metrics()
        return jsonify({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

@app.route('/plot_metrics', methods=['GET'])
def plot_metrics():
    with app.app_context():
        return agent.plot_metrics()


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if username and password are provided
        if not username or not password:
            return render_template('login.html', error='Tên đăng nhập và mật khẩu là bắt buộc.')
        
        ip_address = request.remote_addr

        interaction_data = f"Username: {username}, Password: {password}"

        has_sql_injection = detect_sql_injection(username) or detect_sql_injection(password)
        has_xss = detect_xss(username) or detect_xss(password)

        state = np.array([
            username.isdigit(),
            6 <= len(password) <= 18,
            User.query.filter_by(username=username).first() is not None,
            User.query.filter_by(username=username, password=password).first() is not None,
            has_sql_injection,
            has_xss,
            False  # CSRF detection not applicable
        ], dtype=int).reshape(1, -1)

        action = agent.act(state)
        logging.info(f"Action taken by DQN agent: {action}")
        print(f"Action taken by DQN agent: {action}")  # Debugging print statement
        print(f"State: {state.tolist()}, Action: {action}")  # Debugging print statement

        # Update login attempts tracking
        if username not in login_attempts:
            login_attempts[username] = 0
        login_attempts[username] += 1

        # Check if the IP is blocked
        if ip_address in blocklist:
            logging.warning(f"Địa chỉ IP {ip_address} đã bị chặn khi cố gắng truy cập.")
            return render_template('login.html', error='Access denied.')

        reward = 0.0
        reward_sql = 0.0
        reward_xss = 0.0
        attack_type = None
        true_value_sql = 0.0
        true_value_xss = 0.0

        if action == 1:
            if has_sql_injection:
                attack_type = 'sql_injection'
                reward_sql = 0.004
                true_value_sql = 0.004
                blocklist.add(ip_address)
                response_message = 'Truy cập bị từ chối do phát hiện tấn công SQL Injection.'
                done = True
                logging.info(f"SQL Injection detected, updating metrics")
                agent.update_metrics(attack_type, true_value_sql, action)
                logging.info(f"Updated y_true_sql_injection: {agent.y_true_sql_injection}")
                logging.info(f"Updated y_pred_sql_injection: {agent.y_pred_sql_injection}")

            elif has_xss:
                attack_type = 'xss'
                reward_xss = 0.004
                true_value_xss = 0.004
                blocklist.add(ip_address)
                response_message = 'Truy cập bị từ chối do phát hiện tấn công XSS.'
                done = True
                print(f"XSS detected, updating metrics)")
                logging.info(f"XSS detected, updating metrics")
                agent.update_metrics(attack_type, true_value_xss, action)
                print(f"Updated y_true_xss: {agent.y_true_xss}")
                print(f"Updated y_pred_xss: {agent.y_pred_xss}")
                logging.info(f"Updated y_true_xss: {agent.y_true_xss}")
                logging.info(f"Updated y_pred_xss: {agent.y_pred_xss}")
            else:
                reward_sql = reward_xss = -0.004
                true_value_sql = true_value_xss = 0.0
                response_message = 'Cảnh báo không chính xác.'
                done = True
        else:
            response_message = 'Tương tác bình thường.'
            done = False


        reward = reward_sql if has_sql_injection else (reward_xss if has_xss else 0.0)
        
        next_state = state
        agent.remember(state, action, reward, next_state, done, attack_type)
 
        if agent.memory_size() >= batch_size:
            agent.replay(batch_size)

        if attack_type is not None:
            if attack_type == 'sql_injection':
                agent.update_metrics(attack_type, true_value_sql, action)
            elif attack_type == 'xss':
                agent.update_metrics(attack_type, true_value_xss, action)
        agent.replay(batch_size)
    
        if reward > 0.0:
            return render_template('login.html', error=response_message)
        
        
        if login_attempts[username] >= 15:
            user = User.query.filter_by(username=username).first()
            if user:
                logging.warning(f"Cảnh báo phát hiện nhiều lần đăng nhập khác nhau: {interaction_data}")
              
            blocklist.add(ip_address)  # Block the attacker IP
            return render_template('login.html', error=request.args.get('error'))

        
        if not username.isdigit() or len(username) != 10:
            interaction = Interaction(endpoint='/login', data=interaction_data, response="Đăng nhập thất bại - số điện thoại không hợp lệ ")
            db.session.add(interaction)
            db.session.commit()
            logging.info(f"Đăng nhập thất bại - số điện thoại không hợp lệ : {interaction_data}")
            return render_template('login.html', error='Số điện thoại không hợp lệ') 
        
        if len(password) < 6 and len(password) > 18 :
            interaction = Interaction(endpoint='/login', data=interaction_data, response="Đăng nhập thất bại - Mật khẩu phải từ 6 đến 18 ký tự.")
            db.session.add(interaction)
            db.session.commit()
            logging.info(f"Đăng nhập thất bại - Mật khẩu phải từ 6 đến 18 ký tự : {interaction_data}")
            return render_template('login.html', error='Mật khẩu phải từ 6 đến 18 ký tự.')
        
        user = User.query.filter_by(username=username, password=password).first()
  
        if not user:
            interaction = Interaction(endpoint='/login', data=interaction_data, response="Đăng nhập thất bại - không tìm thấy người dùng")
          
            db.session.add(interaction)
            db.session.commit()
            logging.info(f"Đăng nhập thất bại - không tìm thấy người dùng. {interaction_data}")
            return redirect(url_for('login') + '?error=user_not_found')
        
        interaction = Interaction(endpoint='/login', data=interaction_data, response="Đăng nhập thành công")
        db.session.add(interaction)
        db.session.commit()
        logging.info(f"Đăng nhập thành công: {interaction_data}")
        session['user_id'] = user.id
        reward = 0.0 if not has_sql_injection and not has_xss else -0.004

        next_state = state  # Keep the state same for simplicity
        agent.remember(state, action, reward, next_state, done=False)

        # Check if there are enough samples in the memory to perform a replay
        non_zero_elements = sum(1 for item in agent.memory.tree.data if item is not None and np.any(item != 0))
        if non_zero_elements > batch_size:
            agent.replay(batch_size)

        return redirect(url_for('account_overview'))
    return render_template('login.html', error=request.args.get('error')) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/account_overview')
def account_overview():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = db.session.get(User, session['user_id'])

    return render_template('account_overview.html', user=user)

@app.route('/account')
def account():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = db.session.get(User, session['user_id'])

    return render_template('account.html', user=user)

@app.route('/account_overview/transfer', methods=['GET', 'POST'])
def transfer():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = db.session.get(User, session['user_id'])

    if request.method == 'POST':
        
        try:
            source_stk = user.stk
            recipient_stk = request.form['recipient_account']
            amount = float(request.form['transfer_amount'])
            transfer_detail = request.form['transfer_detail']

            referer = request.headers.get("Referer")
            origin = request.headers.get("Origin")
            has_csrf = detect_csrf(referer, origin)

            state = np.array([
                user.username.isdigit(),
                True,  # Assuming valid password length here
                True,  # Assuming user is logged in
                True,  # Assuming password is correct
                False,  # No SQL Injection in transfer
                False,  # No XSS in transfer
                has_csrf
  
            ], dtype=int).reshape(1, -1)

            action = agent.act(state)
            logging.info(f"Action taken by DQN agent: {action}")
            logging.info(f"State: {state.tolist()}, CSRF Detected: {has_csrf}")
            print(f"Action taken by DQN agent: {action}")
            print(f"State: {state.tolist()}, CSRF Detected: {has_csrf}")

            reward = 0.0
            attack_type = None
            true_value_csrf = 0.0

            if action == 1 and has_csrf:
                logging.warning(f"Detected CSRF in transfer: User ID: {user.id}")
                reward_csrf = 0.004 # Different reward for CSRF
                attack_type = 'csrf'
                true_value_csrf = 0.004
                response_message = 'Transfer failed due to CSRF detection!'
                done = True
            
                agent.replay(batch_size)
                fake_data = generate_fake_data()  # Use fake data if an attack is detected
                return jsonify(fake_data), 200
            
            else:
                reward_csrf = -0.004 # Penalize incorrect alerts
                response_message = 'No CSRF detected.'
                done = True

            reward = reward_csrf if has_csrf else -0.004

            agent.remember(state, action, reward, state, done, attack_type)
            if attack_type == 'csrf':
                agent.update_metrics(attack_type, true_value_csrf, action)
            agent.replay(batch_size)

            if reward == 0.004:
                return jsonify({"message": response_message}), 200

        
            logging.info(f"Yêu cầu chuyển khoản: {source_stk} -> {recipient_stk}, số tiền: {amount}")

            recipient = User.query.filter_by(stk=recipient_stk).first()

            if not recipient:
                logging.warning(f"Người nhận {recipient_stk} không tìm thấy  ")
                return jsonify({"message": "Không tìm thấy người nhận "}), 400
    
            # Convert balance strings to floats for calculations
            user.balance = float(user.balance.replace(',', ''))
            recipient.balance = float(recipient.balance.replace(',', ''))

            if user.balance < amount:
                logging.warning(f"Không đủ số dư để chuyển khoản từ {source_stk} đến {recipient_stk}, số tiền: {amount} ")
                return jsonify({"message": "Không đủ số dư "}), 400

            # Update balances
            user.balance -= amount
            recipient.balance += amount

            # Convert balances back to string with commas
            user.balance = f"{user.balance:,.0f}"
            recipient.balance = f"{recipient.balance:,.0f}"

             # Create transaction record
            transaction = Transaction(
                user_id=user.id,
                recipient_stk=recipient.stk,
                recipient_name=recipient.full_name,
                description=transfer_detail,
                amount=-amount,  # Negative amount for outgoing transactions
                transaction_type='outgoing',
            )
            db.session.add(transaction)

            # Create corresponding incoming transaction record for the recipient
            incoming_transaction = Transaction(
                user_id=recipient.id,
                recipient_stk=user.stk,
                recipient_name=user.full_name,
                description=transfer_detail,
                amount=amount,  # Positive amount for incoming transactions
                transaction_type='incoming',
            )
            db.session.add(incoming_transaction)
            db.session.commit()

            logging.info(f"Chuyển khoản thành công: {source_stk} -> {recipient_stk}, số tiền: {amount}, Nội dung chuyển khoản: {transfer_detail} ")
            flash('Chuyển khoản thành công.')

            # Pop-up message details
            transfer_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"""
            Quý khách đã thực hiện thành công giao dịch chuyển khoản đến số tài khoản thụ hưởng: {recipient_stk}. <br>
            Tên người thụ hưởng: ({recipient.full_name}) . <br>
            số tiền: {amount} VND<br>
            Nội dung chuyển khoản:: {transfer_detail}<br>
            Thời gian giao dịch:: {transfer_time}
            """
             # Add the transition to memory and perform replay
            agent.remember(state, action, reward, state, done=False)
            agent.update_metrics('csrf', true_value_csrf, action)  # Update CSRF metrics

            non_zero_elements = sum(1 for item in agent.memory.tree.data if item is not None and np.any(item != 0))
            
            if non_zero_elements > batch_size:
                agent.replay(batch_size)

                log_evaluation_metrics()
                agent.plot_metrics()
            return jsonify({"message": message}), 200

        except Exception as e:
            logging.error(f"Có lỗi xảy ra trong quá trình chuyển khoản: {str(e)}")
            return jsonify({"message": "Chuyển khoản thất bại!"}), 500
        
    return render_template('transfer.html', user=user)
            
        
@app.route('/transaction_history', methods=['GET'])
def transaction_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = db.session.get(User, session['user_id'])

    #transactions = Transaction.query.filter_by(user_id=user.id).all()
    return render_template('transaction_history.html', user=user) 


@app.route('/search_transactions', methods=['POST'])
def search_transactions():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = db.session.get(User, session['user_id'])

    start_date = request.json.get('start_date')
    end_date = request.json.get('end_date')

    query = Transaction.query.filter(Transaction.user_id == user.id)

    if start_date:
        query = query.filter(Transaction.timestamp >= datetime.strptime(start_date, '%Y-%m-%d'))
    if end_date:
        query = query.filter(Transaction.timestamp <= datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1))

    transactions = query.all()

    transaction_list = []
    for transaction in transactions:
        transaction_list.append({
            'recipient_stk': transaction.recipient_stk,
            'recipient_name': transaction.recipient_name,
            'description': transaction.description,
            'amount': transaction.amount,
            'transaction_type': transaction.transaction_type,  
            'timestamp': transaction.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        })

    return jsonify(transaction_list)

@app.route('/hoidap')
def hoidap():
    return render_template('/html-qa.html')

@app.route('/hoidap/1-dkdn.html')
def dkdn():
    return render_template('/1-dkdn.html')

@app.route('/hoidap/2-mk.html')
def mk():
    return render_template('/2-mk.html')

@app.route('/hoidap/3-chuyen-khoan.html')
def chuyen_khoan():
    return render_template('/3-chuyen-khoan.html')            

@app.route('/hoidap/4-dich-vu-the.html')
def dich_vu_the():
    return render_template('/4-dich-vu-the.html')   

@app.route('/hoidap/5-tin-bien-dong-so-du-ott.html')
def tin_bien_dong_so_du():
    return render_template('/5-tin-bien-dong-so-du-ott.html')   

@app.route('/hoidap/6-cau-hoikhac.html')
def cau_hoi_khac():
    return render_template('/6-cau-hoi-khac.html')   

# Function to open the browser
#def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':

    #threading.Timer(1, open_browser).start()
    app.run(debug=True, host='0.0.0.0', port=5000)


