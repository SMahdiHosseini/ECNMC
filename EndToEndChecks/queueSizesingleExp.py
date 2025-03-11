import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_cdf(data):
        values, base = np.histogram(data, bins=200, density=True)
        cumulative = np.cumsum(values) / np.sum(values)
        return base[:-1], cumulative

def plot_cdf(poisson_df, Q_t_df, q_p_t_df, start, end, parameter):
    x_poisson, y_poisson = compute_cdf(poisson_df[parameter])
    x_Q_t, y_Q_t = compute_cdf(Q_t_df[parameter])
    x_q_p_t, y_q_p_t = compute_cdf(q_p_t_df[parameter])

    plt.figure(figsize=(10, 6))
    plt.plot(x_poisson, y_poisson, label='poisson', color='r')
    plt.plot(x_Q_t, y_Q_t, label='Q(t)', color='b')
    plt.plot(x_q_p_t, y_q_p_t, label='q(p(t))', color='g')

    # plt.axvline(x=poisson_df[parameter].mean(), color='r', linestyle='--', label='Poisson Mean')
    # plt.axvline(x=Q_t_df[parameter].mean(), color='b', linestyle='--', label='Q_t Mean')
    # plt.axvline(x=q_p_t_df[parameter].mean(), color='g', linestyle='--', label='q_p_t Mean')

    plt.xlabel(parameter)
    plt.ylabel('CDF')
    plt.title('CDF of {}'.format(parameter))
    plt.legend()
    plt.grid(True)
    
    plt.savefig(dir + '{}_cdf_{}_to_{}.pdf'.format(parameter, start / 1e9, end / 1e9))
    plt.close()

def plot_queuing_delay(poisson, Q_t, q_p_t):
    # Read CSV files
    poisson_df = pd.read_csv(poisson).sort_values(by='Time')
    Q_t_df = pd.read_csv(Q_t).sort_values(by='Time')
    q_p_t_df = pd.read_csv(q_p_t).sort_values(by='Time')
    
    start = 0.39995 * 1e9
    end = 0.40005 * 1e9
    poisson_df = poisson_df[poisson_df['Time'] >= start]
    poisson_df = poisson_df[poisson_df['Time'] <= end]

    Q_t_df = Q_t_df[Q_t_df['Time'] >= start]
    Q_t_df = Q_t_df[Q_t_df['Time'] <= end]

    q_p_t_df = q_p_t_df[q_p_t_df['Time'] >= start]
    q_p_t_df = q_p_t_df[q_p_t_df['Time'] <= end]

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.scatter(Q_t_df['Time'], Q_t_df['QueuingDelay'], label='Q(t)', color='b', marker='x', s=10)
    plt.scatter(q_p_t_df['Time'], q_p_t_df['QueuingDelay'], label='q(p(t))', color='g', marker='*', s=10)
    plt.scatter(poisson_df['Time'], poisson_df['QueuingDelay'], label='poisson', color='r', marker='+', s=10)
   
    # Labels and title
    plt.xlabel('Time')
    plt.ylabel('Queuing Delay')
    plt.title('Queuing Delay Over Time')
    plt.legend()
    plt.minorticks_on()
    plt.grid(True, alpha=0.5, which='both')
    
    # Save plot
    plt.savefig(dir + 'QueuingDelay_{}_to_{}.pdf'.format(start / 1e9, end / 1e9))
    plt.close()

def plot_queue_sizes(poisson, Q_t, q_p_t):
    # Read CSV files
    poisson_df = pd.read_csv(poisson).sort_values(by='Time')
    Q_t_df = pd.read_csv(Q_t).sort_values(by='Time')
    q_p_t_df = pd.read_csv(q_p_t).sort_values(by='Time')
    
    start = 0.39975 * 1e9
    end = 0.400 * 1e9
    poisson_df = poisson_df[poisson_df['Time'] >= start]
    poisson_df = poisson_df[poisson_df['Time'] <= end]

    Q_t_df = Q_t_df[Q_t_df['Time'] >= start]
    Q_t_df = Q_t_df[Q_t_df['Time'] <= end]

    q_p_t_df = q_p_t_df[q_p_t_df['Time'] >= start]
    q_p_t_df = q_p_t_df[q_p_t_df['Time'] <= end]

    # cdf
    # plot_cdf(poisson_df, Q_t_df, q_p_t_df, start, end, 'QueueSize')
    # plot_cdf(poisson_df, Q_t_df, q_p_t_df, start, end, 'QueuingDelay')
    # plot_cdf(poisson_df, Q_t_df, q_p_t_df, start, end, 'MarkingProb')

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.scatter(Q_t_df['Time'], Q_t_df['QueueSize'], label='Q(t)', color='b', marker='x', s=10)
    plt.scatter(q_p_t_df['Time'], q_p_t_df['QueueSize'], label='q(p(t))', color='g', marker='*', s=10)
    plt.scatter(poisson_df['Time'], poisson_df['QueueSize'], label='poisson', color='r', marker='+', s=10)
   
    # add a horizontal line at y=K
    plt.axhline(y=4500, color='black', linestyle='--', label='K')
    # Labels and title
    plt.xlabel('Time')
    plt.ylabel('Queue Size')
    plt.title('Queue Size Over Time')
    plt.legend()
    plt.minorticks_on()
    plt.grid(True, alpha=0.5, which='both')

    
    # Save plot
    plt.savefig(dir + 'queue_sizes_{}_to_{}.pdf'.format(start / 1e9, end / 1e9))
    plt.close()

def get_time_average(file, column, tag):
    df = pd.read_csv(file)
    # sort the dataframe by 'Time' and 'QueueSize' the lowest time value is the first row, but the rows with the same time value are sorted by 'QueueSize' in ascending order.
    df = df.sort_values(by=['Time', 'QueueSize'], ascending=[True, False])
    time = df['Time'].values
    values = df[column].values
    time_average_left = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
    time_average_right = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
    print('time_average_left of {} for {} is {} and time_average_right is {}'.format(column, tag, time_average_left, time_average_right))

def get_event_average(file, column):
    df = pd.read_csv(file).sort_values(by='Time')
    # calculate the event average of the 'column' over 'Time'. The event average is the sum of the column values divided by the total number of events.
    event_average = df[column].mean()
    print('Event average of {} is {}'.format(column, event_average))
    return event_average

def prepare_e2e_df(e2e_file):
    e2e_df = pd.read_csv(e2e_file)
    e2e_df = e2e_df[e2e_df['SentTime'] != -1]
    e2e_df = e2e_df[e2e_df['IsReceived'] == 1]
    e2e_df['EndToEndDelay'] = e2e_df['ReceiveTime'] - e2e_df['SentTime'] - e2e_df['transmissionDelay']
    e2e_df['SentTime'] = e2e_df['SentTime'] + 50000 + (e2e_df['PayloadSize'] * 8) / 0.474
    e2e_df = e2e_df.sort_values(by='SentTime')

    time = e2e_df['SentTime'].values
    values = e2e_df['EndToEndDelay'].values
    time_average_left = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
    time_average_right = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
    values_linear = (values[1:] + values[:-1]) / 2 
    time_average_lineat = np.sum(values_linear * np.diff(time)) / (time[-1] - time[0])
    print('time_average_left of EndToEndDelay for {} is {} and time_average_right is {} and time_average_lineat is {}'.format('end-to-end measurement', time_average_left, time_average_right, time_average_lineat))

def e2e_marking_prob(e2e_file):
    e2e_df = pd.read_csv(e2e_file)
    e2e_df = e2e_df[e2e_df['SentTime'] != -1]
    e2e_df['ECN'] = e2e_df.apply(lambda x: x['ECN'] if x['IsReceived'] != 0 else 1, axis=1)
    e2e_df['SentTime'] = e2e_df['SentTime'] + 50000 + (e2e_df['PayloadSize'] * 8) / 0.474
    e2e_df = e2e_df.sort_values(by='SentTime')

    time = e2e_df['SentTime'].values
    values = e2e_df['ECN'].values
    time_average_left = np.sum(values[1:] * np.diff(time)) / (time[-1] - time[0])
    time_average_right = np.sum(values[:-1] * np.diff(time)) / (time[-1] - time[0])
    print('time_average_left of marking prob for {} is {} and time_average_right is {}'.format('end-to-end measurement', time_average_left, time_average_right))

# Example usage
dir = '../../Results_forward/0.79/0/'
poisson = dir + "SD0_PoissonSampler_events.csv"
Q_t = dir + "SD0_PoissonSampler_queueSize.csv"
q_p_t = dir + "SD0_PoissonSampler_queueSizeByPackets.csv"
e2e = dir + "A0D0_EndToEnd_packets.csv"

plot_queue_sizes(poisson, Q_t, q_p_t)
plot_queuing_delay(poisson, Q_t, q_p_t)
# get_time_average(q_p_t, 'DropProb', 'Q(P(t))')
# get_time_average(Q_t, 'DropProb', 'Q(t)')
# get_event_average(poisson, 'DropProb')
# print('\n\n\n************\n\n\n')
# get_time_average(q_p_t, 'MarkingProb', 'Q(P(t))')
# get_time_average(Q_t, 'MarkingProb', 'Q(t)')
# e2e_marking_prob(e2e)
# get_event_average(poisson, 'MarkingProb')
# get_event_average(poisson, 'LastMarkingProb')
# print('\n\n\n************\n\n\n')
# get_time_average(q_p_t, 'QueuingDelay', 'Q(P(t))')
prepare_e2e_df(e2e)
# get_time_average(Q_t, 'QueuingDelay', 'Q(t)')
get_event_average(poisson, 'QueuingDelay')
# print('\n\n\n************\n\n\n')


