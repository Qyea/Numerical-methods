from typing import List
from math import fabs
from numpy import arange
import matplotlib.pyplot as plt

import math

class MyRange:
    def __init__(self, a, b):
        if a > b:
            raise ValueError("Invalid range")
        
        self.A = a
        self.B = b
    
    def GetEquidistantKnots(self, n):
        knots = []
        step = (self.B - self.A) / n

        for i in range(n + 1):
            knots.append(self.A + step * i)

        return knots
    
    def GetChebyshevKnots(self, n):
        knots = []

        for i in range(n + 1):
            knots.append((self.A + self.B) / 2 + ((self.B - self.A) / 2) * math.cos(math.pi * (2 * i + 1) / (2 * (n + 1))))

        return knots
    
    def GetPoints(self):
        points = []

        for i in range(101):
            points.append(self.A + i * (self.B - self.A) / 100)

        return points

class Form1:
    def __init__(self):
        self._functions = MyRange(-3, 3)

    def CalculatePn(self, n: int, type: int):
        name = ""
        if type == 1:
            name = "P1"
        elif type == 2:
            name = "P2"
        else:
            raise NotSupportedException()

        result = self.CalculatePolynomial(type, n, 1)
        table = self.GetErrorTable(type, name, 1)

        stringBuilder = ""
        for line in table:
            stringBuilder += line + "\n"

        Pn.Text = stringBuilder

        line1 = LineSeries()
        line1.Title = name
        line1.Color = 'blue'
        line1.StrokeThickness = 1
        line1.MarkerSize = 2
        line1.MarkerType = 'o'

        for item in result.First:
            line1.Points.append(DataPoint(item.First, item.Second))

        return line1

    def CalculateCn(self, n: int, type: int):
        name = ""
        if type == 1:
            name = "C1"
        elif type == 2:
            name = "C2"
        else:
            raise NotSupportedException()

        result = self.CalculatePolynomial(type, n, 2)
        table = self.GetErrorTable(type, name, 2)

        stringBuilder = ""
        for line in table:
            stringBuilder += line + "\n"

        Cn.Text = stringBuilder

        line1 = LineSeries()
        line1.Title = name
        line1.Color = 'red'
        line1.StrokeThickness = 1
        line1.MarkerSize = 2
        line1.MarkerType = 'o'

        for item in result.First:
            line1.Points.append(DataPoint(item.First, item.Second))

        return line1

    def PrintReport(self, n: int, type: int):
        pn = self.CalculatePn(n, type)
        cn = self.CalculateCn(n, type)

        function = None
        if type == 1:
            function = self._functions.F1
        elif type == 2:
            function = self._functions.F2
        else:
            raise NotSupportedException()

        myModel = PlotModel()
        myModel.Title = ""

        legend = Legend()
        legend.LegendTitle = ""
        legend.LegendPosition = LegendPosition.RightTop
        myModel.Legends.append(legend)

        myModel.Series.append(FunctionSeries(function, self._functions.Range.A, self._functions.Range.B, 0.001))
        myModel.Series.append(pn)
        myModel.Series.append(cn)
        plotView1.Model = myModel

    def Button1_Click(self, sender, e):
        if f1.Checked:
            if int.TryParse(textBox1.Text, out var n):
                PrintReport(n, 1)
            else:
                textBox1.Text = "Wrong number!"

        if f2.Checked:
            if int.TryParse(textBox1.Text, out var n):
                PrintReport(n, 2)
            else:
                textBox1.Text = "Wrong number!"

    def F1_CheckedChanged(self, sender, e):
        if f1.Checked:
            f2.Checked = False
        else:
            f2.Checked = True

    def F2_CheckedChanged(self, sender, e):
        if f2.Checked:
            f1.Checked = False
        else:
            f1.Checked = True

    def GetErrorTable(self, type: int, polynomial: str, knotsType: int) -> List[str]:
        result = []

        for i in range(3, 31):
            pair = self.CalculatePolynomial(type, i, knotsType)

            max_val = 0.0

            for j in range(len(pair.First)):
                max_val = max(fabs(pair.First[j].Second - pair.Second[j]), max_val)

            result.append(f"n = {i}: max|{polynomial} - f{type}| = {max_val}")

        return result

    def CalculatePolynomial(self, type: int, n: int, knotsType: int):
        knots = []
        if knotsType == 1:
            knots = self._functions.Range.GetEquidistantKnots(n)
        elif knotsType == 2:
            knots = self._functions.Range.GetChebyshevKnots(n)
        else:
            raise NotSupportedException()

        function = None
        if type == 1:
            function = self._functions.F1
        elif type == 2:
            function = self._functions.F2
        else:
            raise NotSupportedException()
