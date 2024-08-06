import random
import string

def generate_random_phone_number():
    return '0' + ''.join(random.choices(string.digits, k=9))

def generate_random_password():
    chars = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choices(chars, k=16))

def generate_random_stk():
    return ''.join(random.choices(string.digits, k=13))

def generate_vietnamese_name():
    last_names = ['Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng', 'Huỳnh', 'Phan','Vũ','Võ', 'Đặng', 'Bùi', 'Đỗ', 'Hồ', 'Ngô', 'Trịnh', 'Lý', 'Vương', 'Tô', 'Trịnh', 'Cao']
    middle_names = ['Văn', 'Thị', 'Minh', 'Quốc', 'Quang', 'Hồng', 'Hữu', 'Kim', 'Bích', 'Thanh', 'Ngọc', 'Quỳnh', 'Thanh', 'Thiên', 'Hương', 'Kim', 'Mai', 'Phúc', 'Nhật']
    first_names = ['An', 'Bình', 'Châu', 'Dũng', 'Hà', 'Huy', 'Khánh', 'Lan', 'Lệ', 'Linh', 'Mai', 'Minh', 'Ngọc', 'Phương', 'Quân', 'Sơn', 'Thảo', 'Thu', 'Thúy', 'Trang']
    return f"{random.choice(last_names)} {random.choice(middle_names)} {random.choice(first_names)}"

def generate_email(full_name):
    return f"{full_name.lower().replace(' ', '')}@example.com"

users = []
for i in range(50):
    full_name = generate_vietnamese_name()
    users.append(
        {
            'username': generate_random_phone_number(),
            'password': generate_random_password(),
            'balance':  "{:,}".format(random.randint(50000, 100000000)),
            'full_name': full_name,
            'email': generate_email(full_name),
            'stk': generate_random_stk()
        }
    )

for user in users:
    print(user)
