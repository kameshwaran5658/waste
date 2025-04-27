import bcrypt

password = '1510'  # Replace with your real password
hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

print(hashed.decode('utf-8'))
