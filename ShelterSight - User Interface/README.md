<img width="1437" alt="Screenshot 2024-04-24 at 6 30 50 PM" src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/105636722/2f893bee-4825-437b-9fda-5dacf9281ac7">

# Shelter Sight User Interface
## Background
In our journey to create a reliable model, we've dedicated ourselves to crafting a prototype that leverages data collected from the city of Toronto, paired with our models. This user interface (UI) prototype isn't just the final output of our project; it also serves as an illustration of how our data and models can be put into practical use.

This UI prototype represents the culmination of countless hours of research, development, and fine-tuning. It's a tangible manifestation of our commitment to delivering impactful solutions.

## Video Demonstration
The video below provides a demonstration to how the software functions. We will also go over all the functionalities in this readme as well, but this video serves as an alternative way of understanding how ShelterSight works. 

<div align="center">
    <a href="https://www.youtube.com/watch?v=-DDK9pMYjrk">
    <img src="https://img.youtube.com/vi/-DDK9pMYjrk/0.jpg" alt="DEMO VIDEO" style="width:55%;">
    <p>Shelter Sight Demo</p>
  </a>
</div>

## Shelter Selection

In order to view any information about any shelters, you will have to select the program ID of the shelters in the dropdown of Shelter Selection. Once your shelter have been selected, click on the add button. The Selected Shelters count will increment by 1 when the shelter has been added.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/5fa92fd6-c396-4d89-b6c4-238e4fa535aa" width="650" alt="chessBoard">
  <br>
  <em>Figure 1: Shelter 11895 selected and click on add to increment the counter by 1</em>
</p>

## Shelter Information

This section will display the information of the shelters that you have selected. Click on the Program ID's dropdown, and the options available are the Program IDs of all the shelters that you have selected. Once you clicked on any of the Program IDs, its information will show up in the rest of the sections. Information that is viewable in this section incldues the Organization name, Shelter Group, Location Name, Location Address, Postal Code, and Capacity Type. This section is important as the chosen shelter will be the shelter displayed in the "Output" section and the second graph type as well (more information on that in the "Change and View Graph" section.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/a3a0e024-0f5f-4d88-9e78-105d39d69e57" width="650" alt="chessBoard">
  <br>
  <em>Figure 2: Shelter 11895's information being displayed</em>
</p>


## Remove and Reset Functionalities

To remove a shelter that you have chosen in the Shelter Selection section, you can select that shelter in the Shelter Information's dropdown and click the remove button. That will remove the singular shelter that you have selected and decrease the selected shelters count by 1. You also have the option to reset everything by clicking on the reset button. This will clear all selected shelters and the graph and the Forecast Period dates.

## Forecast Period

Forecast Period is a functionality for the user to choose a specific window of time to look at in the graph. When you have chosen your shelters without a forecast period, you can view every bit of data that is available (which includes all the actual data that has come to pass of every shelters and also their predicted data for the next 60 days as well). When you click on Open Calender for the Forecast Period, a popup will appear with two calendars: one for the start of the window, and one for the end. Once the two dates have been selected, you can click on forecast again, and the graph will shrink to the window that you have selected.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/e113a33d-6391-4533-81e1-a393a9b35913" width="650" alt="chessBoard">
  <br>
  <em>Figure 3: Forecast Period of February 1st to August 1st 2024 Displayed on the graph</em>
</p>

## Change and View Graph, and Download Data

This section covers three different functionalities that are related to the graph and data: Change Graph Type, View Graph, and download data. These three functionalties can be activated by clicking on three different buttons with their corresponding name and symbol.

#### Change Graph Type

You can select multiple shelters to view on the graph; therefore, the graph can become populated with too many different shelters. As a result, you have the option to target specific shelters through the Change Graph Type functionalities. You may target specific shelters by clicking on the shelter's Program ID in the Shelter Information dropdown. To target a different shelter, choose their respective Program ID in the same dropdown, then click on Forecast again.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/9affb9b9-3994-4e74-8085-e1af4f6fa74b" width="650" alt="chessBoard">
  <br>
  <em>Figure 4: Shelter 11831 selected as target for Change Graph Type</em>
</p>

#### View Graph

Clicking on the view graph button, the graph that is currently present in the UI will popout. From the popup graph, you can adjust the size of the graph to view it at different dimensions, and you can also download the graph to your local drive.

#### Download Data

Download data will download the actual and predicted data of everysingle chosen shelter into your local drive as a .csv file. 

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/31ed9c26-c442-4b38-ba08-06e14081979c" width="600" alt="chessBoard">
  <br>
  <em>Figure 5: View Graph and Download Data</em>
</p>

## Output

The Output is the section for you to indentify the actual numerical value of the occupancy rate of the chosen shelter. To target a shelter to identify the output, choose its corresponding Program ID in the Shelter Information's dropdown. Once a shelter and date has been selected, the actual numerical data and data type will appear. Data type is an indicator of whether the numerical value that is being displayed is the actual data from that shelter (as in data that has already passed) or predicted data for that shelter from the model.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/d803ebdd-44a8-4b7e-ac05-1aca121a293f" width="600" alt="chessBoard">
  <br>
  <em>Figure 6: Output data for Shelter 11831 on May 1st 2024</em>
</p>

### Select Model

We have implemented different types of LSTM models. You may explore them in the LSTM folder of this repository. You may also choose to see the inferred data from different LSTM implementations. Models included are the Univariate LSTM ([1] jupyter notebook script in the LSTM section), City-wide Multivariate LSTM ([2] jupyter notebook script), and Correlation Grouping LSTM ([3] jupyter notebook script). Every functionalities mentioned above will work the same way, the difference is that the data wil be passed into a different model.
