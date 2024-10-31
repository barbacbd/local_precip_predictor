# Local Precipitation Predictor

The project was created because my wife is constantly wishing for snow. She would prefer to play in the snow for a couple
of days with the kids, and if that isn't possible, at least snow enough to have a delay (not a full cancellation) of school. 
I enjoy surfing, and one of the most talked about climate factors for surf conditions is the El Nino Southern Oscillation (ENSO)
events. While ENSO is believed to be a driving factor for the surf conditions on the East Coast of the US, this is a bit of a misconception.
There are other climate events that have more of an effect on the weather for the East Coast, such as Arctic Oscillation (AO) and 
North Atlantic Oscillation (NAO). 

Can we use any of these indexes to predict the snowfall or if there will be significant local snowfall during the winter months?

# El Nino Southern Oscillation

ENSO is a global climate pattern that emerges from variations in winds and sea surface temperatures over the low latitudes of 
the pacific ocean. It is important to note that this pattern may appear cyclical but it is not predictable. ENSO affects the 
climate of the lower latitudes including the tropics and subtropic regions, and it is connected with other climate patterns that
affect the climate of higher latitudes. 

The warming phase is referred to as El Nino and the cooling phase is referred to as La Nina. El Nino events often occur with higher intensities, but La Nina events typically last longer. 

All values (including discretized event information) for this project are calculated based on NOAA's definiton of events. Each country that monitors 
ENSO cycles has their own threshold for El Nino, Neutral, and La Nina events. All values are on a scale of 0 to 2.0+. The positive numbers indicate El Nino, and, conversely, the negative numbers indicate La Nina. A breakdown of values is shown below

```
Very Strong La Nina: <= -2.0 
     Strong La Nina: -1.9 - -1.5
   Moderate La Nina: -1.4 - -1.0
       Weak La Nina: -0.9 - -0.5
     Not definitive: -0.4 - 0.4
       Weak El Nino: 0.5 - 0.9
   Moderate El Nino: 1.0 - 1.4
     Strong El Nino: 1.5 - 1.9
Very Strong El Nino: >= 2.0
```

# Atlantic Multidecadal Oscillation

The Atlantic Multidecadal Oscillation (AMO) is the theorized variability of the sea surface temperature (SST) of the North Atlantic Ocean on the timescale of several decades.

Originally, AMO was considered for this study, but it was deemed unncessary. AMO is a controversial data source, and there are theories that it may have little to no influence on climate patterns.


# Arctic Oscillation

The Arctic Oscillation (AO) is a back-and-forth shifting of atmospheric pressure between the Arctic and the mid-latitudes of the North Pacific and North Atlantic. When the AO is strongly positive, a strong mid-latitude jet stream steers storms northward, reducing cold air outbreaks in the mid-latitudes. When the AO is strongly negative, a weaker, meandering jet dips farther south, allowing Arctic air to spill into the mid-latitudes.

The AO appeared to be a strong candidate for this particular project, because the local weather data we are studying comes from Virginia Beach in the mid-latitudes. 

Generally the AO alternates between a positive and negative phase.
However, data using a 60-day running mean has implied the oscillation has been trending to more of a positive phase since the 1970s. The AO has trended to a more neutral state in the last decade. The oscillation still fluctuates stochastically between negative and positive values on daily, monthly, seasonal and annual time scales. The large variations over short periods of time makes it very difficult to predict more than 14 days in advance. This means that even though the AO index may appear to be a useful predicitive variable for our model, we cannot predict the prediction variable (and thus it may not be helpful for long term predictions).


# North Atlantic Oscillation

The North Atlantic Oscillation (NAO) is closely related to AO (above). 

Air pressure over two regions drive this oscillation:

- The high latitudes of the North Atlantic Ocean near Greenland and Iceland generally experience lower air pressure than surrounding regions. This zone of low pressure is called the sub-polar low, or sometimes the Icelandic Low.

- Farther to the south, air pressure over the central North Atlantic Ocean is generally higher than surrounding regions. This atmospheric feature is called the subtropical high, or the Azores High.

NOAA describes the NAO indeces with the following information. The NAO is in a positive phase when both the sub-polar low and the subtropical high are stronger than average. During positive NAO phases, the increased difference in pressure between the two regions results in a stronger Atlantic jet stream and a northward shift of the storm track. Consequently, northern Europe experiences increased storminess and precipitation, and warmer-than-average temperatures that are associated with the air masses that arrive from lower latitudes. At the same time, southern Europe experiences decreased storminess and below-average precipitation. In eastern North America, the positive phase of the NAO generally brings higher air pressure, a condition associated with fewer cold-air outbreaks and decreased storminess. The NAO is in a negative phase when both the sub-polar low and the subtropical high are weaker than average. During negative NAO phases, the Atlantic jet stream and storm track have a more west-to-east orientation, and this brings decreased storminess, below-average precipitation, and lower-than-average temperatures to northern Europe. Conversely, southern Europe experiences increased storminess, above-average precipitation, and warmer-than-average temperatures. In eastern North America, the negative phase of NAO generally brings lower air pressure, a condition associated with stronger cold-air outbreaks and increased storminess.

# Historical daily local weather

The local weather for this study was provided by open-meteo.

The following variables were pulled and used during the study:
- temperature_2m_max
- temperature_2m_min
- temperature_2m_mean
- apparent_temperature_max
- apparent_temperature_min
- apparent_temperature_mean
- precipitation_sum
- rain_sum
- snowfall_sum


# Outcomes

## Analyzing Daily Data

The daily data (above) was averaged over each month (for each variable). Then I was able to take the average of the averages and compare it. I decided anything outside of a 10% difference from the average for a particular month was something that needed to be marked.

For instance:

If the average low temperature for 2020 in the month of January is 45 degrees F, and the average low temperature for the January of all years is 50 degrees F, then 2020 had a 5 degree difference which would be outside of the percentage allowed. Thus we are calling this a colder month than normal. The same calculations are made for the precepitation.

## Discretizing outcomes

In general, the snow occurs when there is moisture and the temperatures are present to support the snow (below freezing).

```
                  Warmer
                    |
             2      |       1
    Dry ____________|____________ Wet
                    |
             3      |       4
                    |
                  Colder
```

The (x,y) plot above shows how I would discretize the data to deteremine if it is a wet vs dry month, and if the temperatures are below or above average.

I thought that we would be able to isolate a fair amount of instances where we had colder and wetter months. These months would then provide us with the historical data to be compared to any of the indexes above.

# Outcome

The model has not currently been able to acurately predict if we will have a wet and colder winter. This is probably because the snow storms that we recieve at our lattitude (Virginia Beach) appear to be like perfect storm situations than predictable months or years in advance. It would appear that there may be a slight correlation between the climatic phenonmenon above and the snow events, but it is not definitive at the moment. The AO and NAO indexes were the most correlated with these events, but that is the major issue. If we go above and look, those two indexes are only predictable to 2 weeks in advanced. This supports the fact that we cannot predict this far in advanced. 

My wife will be disappointed, but that is what you get for living in a temperate climate. This also means that we can be more suprised when snow storms do occur. 

To be continued ...