import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.05                   #how large each update step is
starting_slope = -2.5                  #model's first guess for the line slope
starting_intercept = 3.5               #model's first guess for the line intercept
max_training_steps = 120               #how many weight updates to run

rng = np.random.default_rng(0)

#training data: the model sees these inputs and target outputs
#the hidden pattern is roughly target = 1.8 * input - 0.7
input_values = np.linspace(-2, 2, 20)
target_values = 1.8 * input_values - 0.7 + rng.normal(0, 0.25, len(input_values))

current_slope = starting_slope
current_intercept = starting_intercept

slope_history = [current_slope]
intercept_history = [current_intercept]
loss_history = []

for training_step in range(1, max_training_steps + 1):
    #forward pass: use the current weights to make predictions
    predicted_values = current_slope * input_values + current_intercept

    #prediction_errors says how wrong each prediction was
    prediction_errors = predicted_values - target_values

    #loss is the average wrongness across all training examples
    loss = 0.5 * np.mean(prediction_errors ** 2)
    loss_history.append(loss)

    #gradient means "how the loss changes if this weight changes"
    #in a larger model, backpropagation produces these gradients
    gradient_for_slope = np.mean(prediction_errors * input_values)
    gradient_for_intercept = np.mean(prediction_errors)

    #gradient descent: move each weight in the direction that lowers loss
    current_slope = current_slope - learning_rate * gradient_for_slope
    current_intercept = current_intercept - learning_rate * gradient_for_intercept

    slope_history.append(current_slope)
    intercept_history.append(current_intercept)

slope_history = np.array(slope_history)
intercept_history = np.array(intercept_history)
loss_history = np.array(loss_history)

print("slope in      :", starting_slope)
print("intercept in  :", starting_intercept)
print("slope out     :", round(current_slope, 4))
print("intercept out :", round(current_intercept, 4))
print("loss          :", round(loss_history[0], 4), "->", round(loss_history[-1], 4))

#make predictions with the trained line
trained_predictions = current_slope * input_values + current_intercept

#make a loss surface over many possible slope/intercept pairs
slope_grid_values = np.linspace(-4, 4, 200)
intercept_grid_values = np.linspace(-4, 4, 200)
slope_grid, intercept_grid = np.meshgrid(slope_grid_values, intercept_grid_values)

loss_surface = np.zeros_like(slope_grid)

for point_index in range(len(input_values)):
    surface_predictions = slope_grid * input_values[point_index] + intercept_grid
    surface_errors = surface_predictions - target_values[point_index]
    loss_surface += 0.5 * surface_errors ** 2 / len(input_values)

fig, (data_plot, loss_surface_plot, loss_history_plot) = plt.subplots(3, 1, figsize=(7, 14))

data_plot.scatter(input_values, target_values, color="black", label="training data")
data_plot.plot(input_values, trained_predictions, color="crimson", label="trained line")
data_plot.set_xlabel("input value")
data_plot.set_ylabel("target value")
data_plot.set_title("the model learns a line that fits the data")
data_plot.legend()

loss_surface_plot.contour(
    slope_grid,
    intercept_grid,
    loss_surface,
    levels=30,
    cmap="viridis",
    alpha=0.6
)

loss_surface_plot.plot(
    slope_history,
    intercept_history,
    "o-",
    color="crimson",
    ms=3,
    lw=1
)

loss_surface_plot.set_xlabel("slope")
loss_surface_plot.set_ylabel("intercept")
loss_surface_plot.set_title("gradient descent path through weight space")

loss_history_plot.plot(loss_history, "o-", color="crimson", ms=3)
loss_history_plot.set_xlabel("training step")
loss_history_plot.set_ylabel("loss")
loss_history_plot.set_title("loss falls as the line improves")

plt.tight_layout()
plt.show()
