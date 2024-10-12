### COMP1730/6730 project assignment

"""
ANU ID: u7811248
NAME: Timothy Tey
I declare that this submission is my own work
https://www.anu.edu.au/students/academic-skills/academic-integrity
"""
import pandas as pd
import numpy as np

def load_data(filename):
    '''
    This function reads csv file and stores data object as pandas DataFrame

    Parameter:
    - filename (str): Historical data about co2 emissions across countries

    Return:
    - DataFrame: Data structure that contain all information from the CSV file
    '''
    data = pd.read_csv(filename)
    return data


def calculate_average(numerator, denominator):
    '''
    This function calculates the average (or perform division) of two numbers
    Assumption: the denominator will not be zero

    Parameters:
    - numerator (float): a real float
    - denominator (float): a real float
    
    Return:
    - float: result of the division
    '''
    return numerator / denominator


def global_co2_emissions(data, year):
    '''
    This function calculates all co2 emission in million tonnes from the burning of fossil fuels of all countries during a specified year
    Assumption: Any of co2 emission fields is missing or empty will be treated it as zero

    Parameters:
    - data (DataFrame): an object in the same structure as returned from load_data function
    - year (int): an integer, assuming that this year exists in the data

    Return:
    - float: the total of the global co2 emissions in million tonnes during a specified year. 
    '''
    global_co2_data = data
    global_co2_data = global_co2_data[global_co2_data['year'] == year] #Select the data from the specified year

    co2_emission_field = global_co2_data[['cement_co2', 'coal_co2', 'flaring_co2', 'gas_co2'
                                        , 'oil_co2', 'other_industry_co2']].sum().tolist() #Sum each co2 emissions category from selected columns and convert it to a list
    total_co2emission_year = sum(co2_emission_field)

    return total_co2emission_year


def total_co2emission_country(data, year):
    '''
    This function calculates the total co2 emission for each country during a specified year

    Parameters:
    - data (DataFrame): an object in the same structure as returned from load_data function
    - year (int): an integer, assuming that this year exists in the data

    Return:
    - Series: A pandas Series with a dtype of float64, containing total co2 emissions for each country in the specified year 
    '''
    total_emission_country = load_data(data)
    total_emission_country = total_emission_country[total_emission_country['year'] == year]
    total_emission_country = total_emission_country[['cement_co2', 'coal_co2', 'flaring_co2', 
                                                    'gas_co2', 'oil_co2', 'other_industry_co2']].sum(axis=1) #sum the countries' total co2 emission field column by column
    
    return total_emission_country


def top_co2_emitters(data, year, percent):
    '''
    This function identifies the top countries contibute the most to co2 emission and together emitting a certain percentage of co2 compared with the global emissions

    Parameters:
    - data (DataFrame): data that returned from load_data function
    - year (int): an integer that you can assume that this year exists in the data
    - percent (float): a float number between 0 and 100

    Return:
    - List[Tuple[str, float]]: A list of tuples, each containing:
        - country (str): name of the country
        - percentage (float): share of total global co2 emissions for that country during the specified year, expressed as a percentage
    '''
    top_co2_data = load_data(data)
    top_co2_data['total_emission_country'] = total_co2emission_country(data, year) #Sum of each country total co2 emission during that specified year 
    
    emission_of_country = top_co2_data[['country', 'total_emission_country']]
    
    total_emission_country = emission_of_country[['total_emission_country']]
    total_global_emission = global_co2_emissions(data, year) #total of global co2 emission during that specific year

    emission_of_country['emission_country_percentage'] = calculate_average(total_emission_country, total_global_emission) * 100 #calculate the percentage of co2 emissions for each countries in terms of total global co2 emissions
    emission_of_country_inorder = emission_of_country.sort_values(by='emission_country_percentage', ascending=False) #sort countries' co2 emission percentage in descending order
    country_percentage_list = emission_of_country_inorder[['country', 'emission_country_percentage']].values.tolist()

    top_country = []
    cummulative_percentage = 0

    for country, percentage in country_percentage_list: #find the fewest countries that make up the percent of global emissions
        cummulative_percentage += percentage
        top_country.append((country, percentage))
        if cummulative_percentage >= percent:
            break

    return top_country


def co2_and_gdp_rankings(data_with_total_emission, year, questionnumber):
    '''
    This function caclulate co2 emission per capita and gdp per capita for countries in a specific year, and filtered the data to the list based on the question number provided
    Assumption: this function is just for question 2b, and 3

    Parameters:
    - data_with_total_emission (DataFrame): A DataFrame containing total co2 emission for each country, including the population and GDP
    - year (int): an integer, assuming that this year exists in the data
    - questionnnumber (str): A string indicating which data to return:
        - '2b': top_co2_per_person_list
        - '3': top_gdp_per_capita_list
    
    Returns:
    - List[List[str, float, float]]:
        - For '2b': A list of lists of countries ranked by co2 emissions per capita
        - For '3': A list of lists of countries ranked by gdp per capita
    '''
    def data_filtered(dataframe_used):
        '''
        This function filters the data from a DataFrame 

        Parameter:
        - dataframe_used (DataFrame): A pandas DataFrame containing the multiple columns of data

        Return:
        - List[List[str, float, float]]: A list of lists, where each inner list contains:
            - country (str): name of the country
            - co2_per_capita (float): co2 emissions per capita for the country
            - gdp_per_capita (flaot): gdp per capita for that country
        '''
        return dataframe_used[['country', 'co2_per_capita', 'gdp_per_capita']].values.tolist()
    
    co2_per_person = data_with_total_emission[data_with_total_emission['year'] == year]
    co2_per_person['co2_per_capita'] = calculate_average(co2_per_person['total_emission_country'], co2_per_person['population']) * 1000000 #calculate co2 emission per person for each country in tonnes
    co2_per_person['gdp_per_capita'] = calculate_average(co2_per_person['gdp'], co2_per_person['population'])

    if str.lower(questionnumber) == '2b':
        top_co2_per_person = co2_per_person.sort_values(by='co2_per_capita', ascending=False) #sort the data by co2_per_capita for each country in descending order
        top_co2_per_person = top_co2_per_person[['country', 'co2_per_capita', 'gdp_per_capita']] #selected the columns that needed for analysis
        top_co2_per_person_filtered = top_co2_per_person[top_co2_per_person['co2_per_capita'] != 0] #remove the co2_per_capita that equals to zero
        top_co2_per_person_list = data_filtered(top_co2_per_person_filtered)
    
        return top_co2_per_person_list
    
    elif str.lower(questionnumber) == '3':
        top_gdp_per_capita = co2_per_person.sort_values(by='gdp_per_capita', ascending=False) #sort the data by gdp_per_capita for each country in descending order
        top_gdp_per_capita_list = data_filtered(top_gdp_per_capita)

        return top_gdp_per_capita_list
    
    else:
        None


def selected_countries(co2_capita_list, number_of_countries, rich_or_poor):
    '''
    This function select the top number of countries (top co2 emitters per capita) or top number of poorest or richest countries based on the co2_capita_list given
    
    Parameters:
    - co2_capita_list (List of Lists): a list of lists that contain name of the country, co2_per_capita, and gdp_per_capita
    - number_of_countries (int): a positive integers
    - rich_or_poor (str): 

    Returns:
    - List[Tuples[str, float]]: A list of tuples, where each inner list contains:
        - country (str): name of the country
        - co2_per_capita (float): co2 emissions per capita for the country
    '''
    selected_countries = []
    count = 0

    if 'rich' == str.lower(rich_or_poor): #select a k number of rich countries (country and co2_per_capita)
        for element in co2_capita_list:
            country, co2_per_capita, gdp_per_capita = element
            count += 1
            selected_countries.append((country, co2_per_capita))
            if count >= number_of_countries:
                break
    elif 'poor' == str.lower(rich_or_poor): #select a k number of poor countries (country and co2_per_capita)
        co2_capita_list.sort(key=lambda x:x[2]) #sorted by the value gdp_per_capita in ascending order to find the top poorest countries 
        for element in co2_capita_list:
            country, co2_per_capita, gdp_per_capita = element
            count += 1
            selected_countries.append((country, co2_per_capita))
            if count >= number_of_countries:
                break
    else:
        None
    
    return selected_countries


def co2_per_capita(data, year, k):
    '''
    This function identifies the top-k co2 emitters per capita
    Assumption: Ignore those countries do not have any population information during a specified year

    Parameters:
    - data(DataFrame): data that returned from load_data function
    - year(int): an integer, assuming that this year exists in the data
    - k(int): a positive integer 

    Return:
    - List[Tuple[str, float]]: A list of tuples, each containing:
        - country (str): name of the country
        - co2_per_capita (float): co2 emission per capita for that country during the specified year 
    '''
    co2_capita = load_data(data)
    co2_capita['total_emission_country'] = total_co2emission_country(data, year) #total co2 emission for each country during a specified year
    top_co2_capita_list = co2_and_gdp_rankings(co2_capita, year, '2b') #sort the data by co2_per_capita in descending order

    top_countries = selected_countries(top_co2_capita_list, k, 'rich') #find the top-k countries that emitted the most co2
    return top_countries


def co2_rich_vs_poor(data, year, k):
    '''
    This function test a hypothesis that each person in high-income countries tend to emit much larger amount of co2 than the ones in low-income countries.
    Assumption: If the hypothesis is true, we expect that the ratio is higher than 1

    Parameters:
    - data(DataFrame): data that returned from load_data function
    - year(int): an integer, assuming that this year exists in the data
    - k(int): a positive integer 

    Return:
    - float: the fraction of the average CO2 emissions per capita of the k richest countries to that of the k poorest countries (or we say the hypothesis ratio)
    '''
    co2_rich_poor = load_data(data)
    co2_rich_poor['total_emission_country'] = total_co2emission_country(data, year)
    
    top_co2_capita_list = co2_and_gdp_rankings(co2_rich_poor, year, '3') #sorted the data by gdp_per_capita in descending order

    richest_countries = selected_countries(top_co2_capita_list, k, 'rich') #selected top-k richest countries by gdp_per_capita
    poorest_countries = selected_countries(top_co2_capita_list, k, 'poor') #selected top-k poorest countries by gdp_per_capita

    richest_countries_total = 0
    poorest_countries_total = 0

    for i in range(len(richest_countries)): #count the total co2_per_capita for both top-k richest countries and top-k poorest countries
        richest_countries_total += richest_countries[i][1]
        poorest_countries_total += poorest_countries[i][1]
    
    richest_countries_average = calculate_average(richest_countries_total, k)
    poorest_countries_average = calculate_average(poorest_countries_total, k)
    hypothesis_ratio = calculate_average(richest_countries_average, poorest_countries_average)
    
    return hypothesis_ratio


def predict_co2_per_capita(data, year, k):
    '''
    This function predicts co2 emissions for a country in a future year
    Assumption: 
        - Ignore a specified year co2 emissions data if the specified year population data is empty

    Parameters:
    - data (DataFrame): data that returned from load_data function
    - year (int): an integer for a "future" year larger than all years in the data
    - k (int): a positive integer (>=2) for the number of past years to look into.

    Returns:
    - dict{str: float}: a dictionary contains str as keys and float as values
        - keys (str): country name
        - values (float): prediction of co2 emissions per capita (in tonnes) for that country at the future year
    '''
    prediction = data
    result_of_prediction = {country: 0.0 for country in prediction['country'].unique()} #set the every country prediction to 0.0 to prevent any drop of a country during the data filtering

    prediction_filtered = prediction[prediction['population'].notna()] #filter it out the row that contains nan value in population 
    prediction_filtered['total_emission_country'] = prediction_filtered[['cement_co2', 'coal_co2', 'flaring_co2', 
                                                                        'gas_co2', 'oil_co2', 'other_industry_co2']].sum(axis=1) * 1_000_000 #convert to tonnes
    prediction_filtered['emission_per_capita'] = calculate_average(prediction_filtered['total_emission_country'], prediction_filtered['population'])

    for country in prediction_filtered['country'].unique(): #predict the co2 emission country by country
        country_data = prediction_filtered[prediction_filtered['country'] == country]
        country_data = country_data[country_data['year'] <= year].sort_values('year')[-k:] #look back k years historical data
        country_data_reverse = country_data.sort_values('year')[::-1] #sort the data by years in descending order
        
        emission_per_capita = country_data_reverse['emission_per_capita'].values #find the difference between "this year" emission per capita and "last year" emission per capita
        years = country_data_reverse['year'].values
        slopes = np.diff(emission_per_capita) / np.diff(years)
        slopes = np.append(slopes, slopes[-1])
       
        average_slopes = np.mean(slopes)
        average_intercepts = np.mean(emission_per_capita - average_slopes * years)#using average slopes to predict the each years intercept and find the mean of each years intercepts
        
        result_of_prediction[country] = average_slopes * year + average_intercepts

    return result_of_prediction


def uncertainty_co2_per_capita(data, year, k):
    '''
    This function predicts co2 emissions for a country in a future year
    Assumption: 
        - Ignore a specified year co2 emissions data if the specified year population data is empty
        - k value must be larger than 2

    Parameters:
    - data (DataFrame): data that returned from load_data function
    - year (int): an integer for a "future" year larger than all years in the data
    - k (int): a positive integer (>=2) for the number of past years to look into.

    Returns:
    - dict{str: float}: a dictionary contains str as keys and float as values
        - keys (str): country name
        - values (float): prediction of co2 emissions per capita (in tonnes) for that country at the future year
    '''
    uncertainty = load_data(data)
    uncertainty_prediction = {country: (0.0, 0.0, 0.0) for country in uncertainty['country'].unique()} 
    prediction_data = {country: [] for country in uncertainty['country'].unique()}

    while k >= 2:
        prediction_dict = predict_co2_per_capita(uncertainty, year, k)
            
        for country, co2_per_capita_prediction in prediction_dict.items():
            prediction_data[country].append(co2_per_capita_prediction)
        k -= 1
        
    #prediction_dataframe = pd.DataFrame(prediction_data).T

    #prediction_dataframe['median'] = prediction_dataframe.median(axis=1)
    #prediction_dataframe['percentile_10'] = prediction_dataframe.quantile(0.1, axis=1)
    #prediction_dataframe['percentile_90'] = prediction_dataframe.quantile(0.9, axis=1)

    for country in uncertainty['country'].unique():
        for country, prediction in prediction_data.items():
            prediction = np.array(prediction, dtype=np.float64)
            median = np.median(prediction)
            percentile_10 = np.percentile(prediction, 10)
            percentile_90 = np.percentile(prediction, 90)
            uncertainty_prediction[country] = median, percentile_10, percentile_90
    
    return uncertainty_prediction

print(uncertainty_co2_per_capita('owid-co2-data.csv', 2024, 10))





