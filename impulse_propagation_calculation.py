import pandas as pd
import numpy
import sys
import multiprocessing
from numba import jit
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from time import sleep
from os import system, name

@jit(nopython=True)
def dong2012(x, D):
    # Creating a list of calculated water level at every time step.
    calculated_WL = []
    # Parameter D1 is needed for the calculation and is dependent on x (thickness) and D (diffusivity values)
    D1 = x*(numpy.sqrt(1 / D))

    # Calculation of water level response at every time step (Dong et.al, 2012)
    i = ntimes
    while i <= n:
        h = 0
        j = i-ntimes
        while j < i:
            h = h + (reference_data[j] * ((time[i] - time[j]) ** -1.5) * numpy.exp(-x ** 2 / (4 * D * (time[i] - time[j]))) * dt)
            j += 1

        head = (D1 / F) * h
        calculated_WL.append(head)
        i += 1

    # Calculating the MSE of the calculated and measured water levels
    summation = 0
    for i in range(ntimes, n):  # 1008 is 7 days worth of measuring with a 10 min time step. Can be changed if needed.
        measured = measured_WL[i]
        calculated = calculated_WL[i - ntimes]
        difference = measured - calculated
        difference_squared = difference ** 2
        summation = summation + difference_squared

    MSE = summation / (n - ntimes)  # 1008 is 7 days worth of measuring with a 10 min time step. Can be changed if needed.
    RMSE = numpy.sqrt(MSE)

    return x, D, MSE, RMSE

def ask_detrending():

    answer = ""

    while answer not in ["Y", "N"]:
        answer = input("Does the data need detrending? \n Type Y for yes and N for no: ").upper()

        if answer not in ["Y", "N"]:
            print("Incorrect input! Please type Y or N")

    return answer

def ask_firstplot():

    answer = ""

    while answer not in ["Y", "N"]:
        answer = input("Do you want to detrend the first timeseries? \n Type Y for yes and N for no: ").upper()

        if answer not in ["Y", "N"]:
            print("Incorrect input! Please type Y or N")

    return answer

def ask_secondplot():

    answer = ""

    while answer not in ["Y", "N"]:
        answer = input("Do you want to detrend the second timeseries? \n Type Y for yes and N for no: ").upper()

        if answer not in ["Y", "N"]:
            print("Incorrect input! Please type Y or N")

    return answer

def ask_degree():

    answer = ""

    while answer.isdigit() == False:

        answer = input("Please write the degree of polynom for detrending.\nType Write 1 if the trend is linear: ")

        if answer.isdigit() == False:
            print("Incorrect input! Please type a number")

    return int(answer)

def ask_continue():
    answer = ""

    while answer not in ["Y", "N"]:
        answer = input("Are you happy with this detrending \n"
                       "Type Y for yes and N for no: ").upper()

        if answer not in ["Y", "N"]:
            print("Incorrect input! Please type Y or N")

    return answer

def clear():
    # for windows the name is 'nt'
    if name == 'nt':
        _ = system('cls')

    # and for mac and linux, the os.name is 'posix'
    else:
        _ = system('clear')

# Start of the impulse propagation and vertical Diffusivity calculations
if __name__ == '__main__':

    # DEFINING THE VARIABLES (INPUT IS NEEDED HERE!)
    ###################################################################################################################
    ###################################################################################################################

    #Define value or range of values for peat thickness (x) and diffusivity (D) respectively.
    x_values = 0.4 # The thickness of peat between two sensors
    D_values = numpy.linspace(0.1, 100, 300, dtype=float) # array length must be > 300!
    Dvalues_length = len(D_values - 1)

    for i in range(1, 100):
        D_values[i] = D_values[0] + i * 0.1
    for i in range(100, 200):
        D_values[i] = D_values[99] + (i - 99) * 0.5
    for i in range(200, 300):
        D_values[i] = D_values[199] + (i - 199) * 1.0
    for i in range(300, Dvalues_length):
        D_values[i] = D_values[299] + (i - 299) * 5.0

    # FROM HERE THE DATA IS READ INTO THE PROGRAM.
    # Write here the dataset name (make sure that the file name and sheet name inside it
    # are the same in the dir folder). The files should be in .xlsx format. All files should be places in dir folder.
    # See template files for file structure and column headers. 
    sheetname = "template_file" # reference data location/file
    sheetname2 = "template_file_2" # measured data location/file

    # The file with the calculated values. Write an appropriate file name.
    filename = "output"

    # Defining the period of data removal from the beginning of the time-series and choosing the calculating period for
    # the response function at each time-step (looking back in the data for n days).
    measuring_interval = 10  # NB!! change this parameter NB! IN MINUTES. The measuring interval of the data
    period_1 = 7 # How many days of data to remove from the beginning of the time series (if no need then 0).
    period_2 = 7 # The calculation period for the response function on each time step (days).


    # NB!! DON'T CHANGE THESE PARAMETERS
    interval = 24 * 60 / measuring_interval  # measuring intervals in a day.
    print(interval)
    ntimes = int(period_2 * interval)  # period with what to calculate the response function.
    ntimes2 = int(period_1 * interval) # period removed from the beginning of the time_series (so called dirty data due to
    # equilibration/pressure impulses dissipation of data after pushing in the piezometer).


# THE RUN OF THE PROGRAM STARTS HERE (NO NEED TO CHANGE ANYTHING FROM HERE)
########################################################################################################################
########################################################################################################################
    scaler = StandardScaler(with_std=False)

    # Importing pressure-time data
    dataframe = pd.ExcelFile(sheetname + ".xlsx")
    data = dataframe.parse(sheetname)  # Check the name of your worksheet in excel
    df = pd.DataFrame(data)

    dataframe2 = pd.ExcelFile(sheetname2 + ".xlsx")
    data2 = dataframe2.parse(sheetname2)  # Check the name of your worksheet in excel
    df2 = pd.DataFrame(data2)

    # Visualising data and removing trends if necessary
    X1 = df["time_(days)"].to_numpy()
    X1 = X1[ntimes2:len(X1)]
    X1_new = X1

    Y1 = df["pressure"].to_numpy()
    Y1 = Y1[ntimes2:len(Y1)]
    Y1_new = Y1

    X2 = df2["time_(days)"].to_numpy()
    X2 = X2[ntimes2:len(X2)]
    X2_new = X2
    #print(X2)

    Y2 = df2["pressure"].to_numpy()
    Y2 = Y2[ntimes2:len(Y2)]
    Y2_new = Y2
    #print(Y2)

    # Creating 2D arrays from the X values.
    X1 = numpy.reshape(X1, (len(X1), 1))
    X2 = numpy.reshape(X2, (len(X2), 1))

    # Visualising measured data to check for the need of detrending.
    # NB!!! every plot that pops up needs to be closed for the code to continue.
    plt.plot(X1, Y1)
    plt.plot(X2, Y2)
    plt.legend([sheetname, sheetname2])
    plt.show()

    Y1 = Y1.reshape((len(Y1), 1))
    scaled_Y1_raw = scaler.fit_transform(Y1)
    scaled_Y1_list = [j for i in scaled_Y1_raw for j in i]
    scaled_Y1 = numpy.array(scaled_Y1_list)

    Y2 = Y2.reshape((len(Y2), 1))
    scaled_Y2_raw = scaler.fit_transform(Y2)
    scaled_Y2_list = [j for i in scaled_Y2_raw for j in i]
    scaled_Y2 = numpy.array(scaled_Y2_list)


    detrend = ask_detrending()
    clear()


    if detrend == "Y":

        continue_detrending = True

        while continue_detrending:

            first_detrend = False
            second_detrend = False
            firstplot = ask_firstplot()
            clear()

            if firstplot == "Y":

                i = ask_degree()
                clear()
                pf = PolynomialFeatures(degree=i)
                Xp = pf.fit_transform(X1)
                model1 = LinearRegression()
                model1.fit(Xp, scaled_Y1)
                trendp = model1.predict(Xp)
                detrpoly = [scaled_Y1[j] - trendp[j] for j in range(0, len(Y1))]
                time = X1
                first_detrend = True

            secondplot = ask_secondplot()
            clear()

            if secondplot == "Y":

                i2 = ask_degree()
                clear()
                pf = PolynomialFeatures(degree=i2)
                Xp2 = pf.fit_transform(X2)
                model2 = LinearRegression()
                model2.fit(Xp2, scaled_Y2)
                trendp2 = model2.predict(Xp2)
                detrpoly2 = [scaled_Y2[j] - trendp2[j] for j in range(0, len(Y2))]
                time = X2
                second_detrend = True


            # Plotting the trending results
            if first_detrend and second_detrend:
                fig, axs = plt.subplots(2, 2)

                axs[0, 0].plot(X1, scaled_Y1)
                axs[0, 0].plot(X1, trendp)
                axs[0, 0].set_title(f'Plot ({sheetname})  Poly_degree {i}')
                axs[0, 0].legend(["Data", "Trendline"])

                axs[0, 1].plot(X2, scaled_Y2)
                axs[0, 1].plot(X2, trendp2)
                axs[0, 1].set_title(f'Plot ({sheetname2})  Poly_degree {i2}')
                axs[0, 1].legend(["Data", "Trendline"])

                axs[1, 0].plot(X1, scaled_Y1)
                axs[1, 0].plot(X2, scaled_Y2)
                axs[1, 0].set_title("Before detrending")
                axs[1, 0].legend([sheetname, sheetname2])

                axs[1, 1].plot(X1, detrpoly)
                axs[1, 1].plot(X2, detrpoly2)
                axs[1, 1].set_title("After detrending")
                axs[1, 1].legend([sheetname, sheetname2])

                for ax in axs.flat:
                    ax.set(xlabel="Time (days)", ylabel="Water level (mH2O)")

                plt.tight_layout()
                plt.show()

                reference_data = numpy.array(detrpoly)
                measured_WL = numpy.array(detrpoly2)

            if first_detrend and not second_detrend:
                fig, axs = plt.subplots(2, 2)

                axs[0, 0].plot(X1, scaled_Y1)
                axs[0, 0].plot(X1, trendp)
                axs[0, 0].set_title(f'Plot ({sheetname})  Poly_degree {i}')
                axs[0, 0].legend(["Data", "Trendline"])

                axs[0, 1].plot(X2, scaled_Y2)
                axs[0, 1].set_title(f'Plot ({sheetname2})')
                axs[0, 1].legend(["Data"])

                axs[1, 0].plot(X1, scaled_Y1)
                axs[1, 0].plot(X2, scaled_Y2)
                axs[1, 0].set_title("Before detrending")
                axs[1, 0].legend([sheetname, sheetname2])

                axs[1, 1].plot(X1, detrpoly)
                axs[1, 1].plot(X2, scaled_Y2)
                axs[1, 1].set_title("After detrending")
                axs[1, 1].legend([sheetname, sheetname2])

                for ax in axs.flat:
                    ax.set(xlabel="Time (days)", ylabel="Water level (mH2O)")

                plt.tight_layout()
                plt.show()

                reference_data = numpy.array(detrpoly)
                measured_WL = numpy.array(scaled_Y2)

            if second_detrend and not first_detrend:
                fig, axs = plt.subplots(2, 2)

                axs[0, 0].plot(X1, scaled_Y1)
                axs[0, 0].set_title(f'Plot ({sheetname})')
                axs[0, 0].legend(["Data", "Trendline"])

                axs[0, 1].plot(X2, scaled_Y2)
                axs[0, 1].plot(X1, trendp2)
                axs[0, 1].set_title(f'Plot ({sheetname2}) Poly_degree {i2}')
                axs[0, 1].legend(["Data"])

                axs[1, 0].plot(X1, scaled_Y1)
                axs[1, 0].plot(X2, scaled_Y2)
                axs[1, 0].set_title("Before detrending")
                axs[1, 0].legend([sheetname, sheetname2])

                axs[1, 1].plot(X1, scaled_Y1)
                axs[1, 1].plot(X2, detrpoly2)
                axs[1, 1].set_title("After detrending")
                axs[1, 1].legend([sheetname, sheetname2])

                for ax in axs.flat:
                    ax.set(xlabel="Time (days)", ylabel="Water level (mH2O)")

                plt.tight_layout()
                plt.show()

                reference_data = numpy.array(scaled_Y1)
                measured_WL = numpy.array(detrpoly2)

            continue_workflow = ask_continue()
            clear()

            if continue_workflow == "N":
                continue

            elif ((continue_workflow == "Y") and (not first_detrend) and (not second_detrend)):
                print("Detrending sequence was not started!")
                reference_data = numpy.array(scaled_Y1)
                measured_WL = numpy.array(scaled_Y2)
                continue_detrending = False
                sleep(5)

            else:
                continue_detrending = False



    else:
        print("No need for detrending! Proceeding to calculations.")
        reference_data = numpy.array(scaled_Y1)
        measured_WL = numpy.array(scaled_Y2)
        #reference_data = Y1_new
        #measured_WL = Y2_new
        sleep(5)

    clear()

    print("Starting the calculations")

    # Needed for later
    #time = df["Time (days)"].to_numpy()
    time = X1_new
    print(time)
    #reference_data = df["water_level"].to_numpy()

    # Needed for later
    #time_1 = df2["Time (days)"].to_numpy()
    time_1 = X2_new
    #measured_WL = df2["water_level"].to_numpy()


    # Defining constant parameters
    n = len(time) - 1
    dt = time[2] - time[1]
    F = 2 * numpy.sqrt(numpy.pi)


    # Creating a 2D array of the x and D values
    Diffusivity = []
    thickness = []
    for x in x_values:
        for D in D_values:
            thickness.append(x)
            Diffusivity.append(D)
    Dataset = numpy.vstack((thickness, Diffusivity)).T



    # Choose the amount of processors which are used for calculations (multiprocessing). The ammount should be less
    # than total threads of your PC processor.
    processors = 2

    # Starting the calculation
    p = multiprocessing.Pool(processors)
    results = p.starmap(dong2012, Dataset)
    datatable = pd.DataFrame(results)
    datatable.columns = ["x", "D", "MSE", "RMSE"]

    #Writing the results to a csv file and getting the min D and x values based on the MSE values.
    datatable.to_csv("MSE_" + filename + ".csv")
    mindiffusivity = datatable.loc[datatable["MSE"].idxmin()]["D"]
    minthickness = datatable.loc[datatable["MSE"].idxmin()]["x"]

    #exporting calculated_WL, measured_WL to a txt file with corresponding min x and min D values.
    sys.stdout = open(filename + ".txt", "w")
    print(mindiffusivity)
    print(minthickness)
    D = mindiffusivity
    x = minthickness
    D1 = x * (numpy.sqrt(1 / D))

    # Calculation of water level response with x and D values according to minimum MSE value.
    i = ntimes
    calculated_WL = []
    while i <= n:
        h = 0
        j = i - ntimes
        while j < i:
            h = h + (reference_data[j] * ((time[i] - time[j]) ** -1.5) * numpy.exp(
                -x ** 2 / (4 * D * (time[i] - time[j]))) * dt)
            j += 1
        head = (D1 / F) * h
        calculated_WL.append(head)
        print(i, time[i], reference_data[i], measured_WL[i], calculated_WL[i - ntimes])
        i += 1

    sys.stdout.close()


    p.close()
    p.join()
