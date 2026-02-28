import bcrypt

# The bcrypt hash (from your example)
bcrypt_hash = b"$2b$12$Nf9LUxtpABWhasLniBLaFO9n3ZFGOAdTNOUIek1qHw8Yizm8mNfMu"

# The password to verify
password = "Anwesh"

# Check if the password matches the hash
if bcrypt.checkpw(password, bcrypt_hash):
    print("Password matches the hash!")
else:
    print("Password does not match the hash.")