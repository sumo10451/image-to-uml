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

Sub ExtractConnections()
    Dim pg As Visio.Page
    Set pg = ActivePage
    Dim shp As Visio.Shape
    Dim connector As Visio.Shape
    Dim fromShape As Visio.Shape
    Dim toShape As Visio.Shape
    Dim output As String
    
    output = "Service Connections:" & vbCrLf & vbCrLf
    
    For Each connector In pg.Shapes
        If connector.OneD = True Then ' Check if the shape is a connector
            If connector.Connects.Count >= 2 Then
                Set fromShape = connector.Connects.Item(1).ToSheet
                Set toShape = connector.Connects.Item(2).ToSheet
                
                ' Get shape names and any custom properties
                Dim fromName As String
                Dim toName As String
                fromName = fromShape.Name
                toName = toShape.Name
                
                ' Optionally, retrieve custom properties (Shape Data)
                Dim fromServiceName As String
                Dim toServiceName As String
                
                If fromShape.CellExistsU("Prop.ServiceName", False) Then
                    fromServiceName = fromShape.CellsU("Prop.ServiceName").ResultStr("")
                Else
                    fromServiceName = "N/A"
                End If
                
                If toShape.CellExistsU("Prop.ServiceName", False) Then
                    toServiceName = toShape.CellsU("Prop.ServiceName").ResultStr("")
                Else
                    toServiceName = "N/A"
                End If
                
                ' Append to output string
                output = output & "Connection from " & fromName & " (Service: " & fromServiceName & ") to " & toName & " (Service: " & toServiceName & ")" & vbCrLf
            End If
        End If
    Next connector
    
    ' Display the results
    MsgBox output
End Sub