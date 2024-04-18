import numpy as np

def generate_channel_gain_matrix_2(channel_type, matrix_shape, channel_snr, K=None):
    if channel_type == "Rayleigh":
        gain_data = np.random.normal(size=matrix_shape) + 1j * np.random.normal(size=matrix_shape)
    elif channel_type == "Rician":
        if K is None:
            raise ValueError("`K` must be provided for Rician channel.")
        direct_component = np.sqrt(K) * np.random.normal(size=matrix_shape)
        scatter_component = np.sqrt(1 / K) * np.random.normal(size=matrix_shape)
        gain_data = direct_component + scatter_component
    elif channel_type == "Gaussian":
        gain_data = np.random.normal(size=matrix_shape)
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")
    magnitudes = np.abs(gain_data)
    normalized_gain_data = (np.sqrt(channel_snr) * magnitudes) / np.linalg.norm(magnitudes)

    return normalized_gain_data


def generate_channel_gain_matrix(channel_type, matrix_size, snr=0, rician_k=None):
    noise_power = 1 / (10 ** (snr / 10))

    if channel_type == "Rayleigh":
        real_part = np.random.normal(0, np.sqrt(1 / 2), matrix_size) * np.sqrt(noise_power)
        imaginary_part = np.random.normal(0, np.sqrt(1 / 2), matrix_size) * np.sqrt(noise_power)
        channel_gain_matrix = real_part + 1j * imaginary_part
    elif channel_type == "Rician":
        if rician_k is None:
            raise ValueError("Rician K-factor required for the Rician channel")

        sigma = np.sqrt(1 / (1 + rician_k)) * np.sqrt(noise_power)
        mu = np.sqrt(rician_k * sigma ** 2)

        real_part = np.random.normal(mu, sigma, matrix_size)
        imaginary_part = np.random.normal(mu, sigma, matrix_size)
        channel_gain_matrix = real_part + 1j * imaginary_part
    elif channel_type == "Gaussian":
        real_part = np.random.normal(0, np.sqrt(noise_power), matrix_size)
        channel_gain_matrix = real_part
    else:
        raise ValueError("Unsupported channel type")

    return channel_gain_matrix

def generate_real_noise(shape, snr):
    noise_power = 1 / (10 ** (snr / 10))
    noise = np.random.normal(0, np.sqrt(1), shape) * np.sqrt(noise_power)
    return noise

def transmit_data(matrix, channel_gain_matrix, snr):
    received_signal = np.dot(matrix, channel_gain_matrix)
    noise = generate_real_noise(received_signal.shape, snr)
    return received_signal + noise

def zero_forcing_equalizer(received_data, channel_gain_matrix):
    inverse_matrix = np.linalg.pinv(channel_gain_matrix)
    return np.dot(received_data, inverse_matrix)

def main():
    data_size = (1, 5)
    matrix_size = (5, 5)
    snr = 25

    data_matrix = np.random.uniform(low=-1, high=1, size=data_size)

    # Rayleigh channel
    rayleigh_gain_matrix = generate_channel_gain_matrix_2("Rayleigh", matrix_size, snr)
    print(rayleigh_gain_matrix)
    rayleigh_transmitted_data = transmit_data(data_matrix, rayleigh_gain_matrix, snr)
    rayleigh_recovered_data = zero_forcing_equalizer(rayleigh_transmitted_data, rayleigh_gain_matrix)

    # Rician channel
    rician_k = 3
    rician_gain_matrix = generate_channel_gain_matrix_2("Rician", matrix_size, snr, rician_k)
    rician_transmitted_data = transmit_data(data_matrix, rician_gain_matrix, snr)
    rician_recovered_data = zero_forcing_equalizer(rician_transmitted_data, rician_gain_matrix)

    # Gaussian channel
    gaussian_gain_matrix = generate_channel_gain_matrix_2("Gaussian", matrix_size, snr)
    gaussian_transmitted_data = transmit_data(data_matrix, gaussian_gain_matrix, snr)
    gaussian_recovered_data = zero_forcing_equalizer(gaussian_transmitted_data, gaussian_gain_matrix)
    print("Raw Data:")
    print(data_matrix)

    print("Rayleigh Recovered Data:")
    print(rayleigh_recovered_data)

    print("Rician Recovered Data:")
    print(rician_recovered_data)

    print("Gaussian Recovered Data:")
    print(gaussian_recovered_data)

if __name__ == "__main__":
    main()


