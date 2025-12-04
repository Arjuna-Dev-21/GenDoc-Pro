import datetime

class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        self.transaction_history = []

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            self.log_transaction("Deposit", amount)
            return True
        return False

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            self.log_transaction("Withdrawal", -amount)
            return True
        return False

    def log_transaction(self, type, amount):
        timestamp = datetime.datetime.now()
        self.transaction_history.append({
            "type": type,
            "amount": amount,
            "time": timestamp
        })

def process_transaction(account, transaction_type, amount):
    """
    Helper function to process transactions safely.
    """
    print(f"Processing {transaction_type} for {account.owner}...")
    
    if transaction_type == "deposit":
        success = account.deposit(amount)
    elif transaction_type == "withdraw":
        success = account.withdraw(amount)
    else:
        print("Invalid transaction type")
        return

    if success:
        print(f"Success! New Balance: {account.balance}")
    else:
        print("Transaction failed (Invalid amount or insufficient funds)")

# Example Usage
if __name__ == "__main__":
    my_account = BankAccount("Alice", 100)
    process_transaction(my_account, "deposit", 50)
    process_transaction(my_account, "withdraw", 200) # Should fail
    process_transaction(my_account, "withdraw", 30)  # Should succeed