Sub ExtractConnections()
    Dim pg As Visio.Page
    Set pg = ActivePage
    Dim shp As Visio.Shape
    Dim connector As Visio.Shape
    Dim fromShape As Visio.Shape
    Dim toShape As Visio.Shape

    For Each connector In pg.Shapes
        If connector.OneD = True Then ' Check if the shape is a connector
            If connector.Connects.Count >= 2 Then
                Set fromShape = connector.Connects.Item(1).ToSheet
                Set toShape = connector.Connects.Item(2).ToSheet

                Debug.Print "Connection from " & fromShape.Name & " to " & toShape.Name
            End If
        End If
    Next connector
End Sub