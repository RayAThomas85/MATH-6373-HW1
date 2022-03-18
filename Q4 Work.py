# Trying combinations of different h values and BATS
import time

h_vals = [23, 3*23]
BATS = [len(TRN/40), len(TRN/20), len(TRN/10), len(TRN/2)]
TOTEP = [100, 150]

for i in h_vals:
  for j in BATS:
    time1 = time.time()
    model = Sequential()
    model.add(Dense(i, input_shape=(35,), activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()

    model.complie(loss='mse', optimizer='adam', metrics=['mse'])
    history = model.fit(X_train, y_train, epochs= TOTEP[1], batch_size = j, verbose=1, validation_split=0.2)
    time2 = time.time()
    print("Computing Time for h = ", i, "and BAT = ", j, ": ", time2 - time1)
    print()
    print("Number of batches: ", j)