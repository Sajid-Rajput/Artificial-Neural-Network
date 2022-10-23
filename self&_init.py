class Employee:
    def __init__(self, work_position, work_on):
        self.position = work_position
        self.working = work_on
        

    def printdetails(self):
        return f"My work position is {self.position} and my working is on {self.working}"


sajid = Employee("Sr. Developer", "Python Djanago Backend")
hamza = Employee("Jr. Developer", "JavaScript Developer")

# sajid.position = "Sr. Developer"
# sajid.working = "Python Djanago Backend"

# hamza.position = "Jr. Developer"
# hamza.working = "JavaScript Developer"

print(sajid.printdetails())
print(hamza.printdetails())

# print(sajid.position, sajid.working)
# print(hamza.position, hamza.working)

# print(sajid.__dict__)
# print(hamza.__dict__)
