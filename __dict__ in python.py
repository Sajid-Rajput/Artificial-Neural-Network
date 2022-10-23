class Employee():
    leaves = 8

sajid = Employee()
hamza = Employee()

sajid.position = "Sr. Developer"
sajid.working = "Python Djanago Backend"

hamza.position = "Jr. Developer"
hamza.working = "JavaScript Developer"
hamza.leaves = 10

# Here, we changing the number of leaves using Employee() class
Employee.leaves = 20 

print(sajid.__dict__)
print(hamza.__dict__)

print(sajid.leaves, hamza.leaves)

