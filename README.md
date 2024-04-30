# Optimization and Analytics Coursework

This main topics of this course were:
* Introduction to the modeling process in decision-making problems. This will include an overview of the different types of optimization and simulation models, as well as the steps involved in the modeling process.
* Linear models: modeling, applications, and the simplex method. This will cover how to formulate linear programming problems, how to solve them using the simplex method, and how to interpret the results.
* Discrete models: applications, binary variables, logical constraints, and algorithms. This will cover how to formulate discrete optimization problems, how to solve them using various algorithms, and how to interpret the results.
* Nonlinear models: applications, optimality conditions, and machine learning algorithms. This will cover how to formulate nonlinear programming problems, how to solve them using various algorithms, and how to interpret the results.
* Case studies. This will include real-world examples of how optimization and simulation models have been used to solve business problems.

Unfortunetly I am only able to upload the first case study for privacy reasons.

## Case study 1

During this case study, we were asked to find a dataset, in our case we selected a dataset based on car sharing data, then we transformed it so we would obtain a demand for 5 sectors we created 
and for the 7 days of the week, then we created a model in pyomo in order to optimize the prices depending on the demand. For it we created a set of constraints 
based on meeting a certain demand and achiving a minimun facturation. 

Also, in order to add binary constraints, we added the constraint that each car should pass through maintence with a certain frequency every week.

I really encourage you to take a look at `code.ipynb` since it is really well explained.
