import pickle

model = pickle.load(open("model.pkl", "rb"))

study = float(input("Study Hours: "))
social = float(input("Social Media Hours: "))
sleep = float(input("Sleep Hours: "))
attendance = float(input("Attendance: "))
assignment = float(input("Assignment Score: "))

prediction = model.predict([[study, social, sleep, attendance, assignment]])

print("Predicted Final Grade:", prediction[0])