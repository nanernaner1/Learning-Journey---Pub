Me:
Can WinSDK insert screenshots into Toast notifications?

Bing:

Yes, you can include images in Toast notifications using the Windows SDK. However, the image must be available as a file or a URL, it cannot be directly inserted as a bitmap or other in-memory format¹.

Here's an example of how to include an image in a Toast notification:

```python
import winrt.windows.ui.notifications as notifications
import winrt.windows.data.xml.dom as dom

nManager = notifications.ToastNotificationManager
notifier = nManager.create_toast_notifier(r"C:\\Path\\To\\Your\\App.exe")

tString = """
<toast>
  <visual>
    <binding template='ToastGeneric'>
      <text>Message from Your App!</text>
      <text>Hi there!</text>
      <image placement="appLogoOverride" hint-crop="circle" src="https://path/to/your/image.jpg"/>
    </binding>
  </visual>
</toast>
"""

xDoc = dom.XmlDocument()
xDoc.load_xml(tString)

notification = notifications.ToastNotification(xDoc)

# Display notification
notifier.show(notification)
```

In this code, replace `"https://path/to/your/image.jpg"` with the URL or local path of your screenshot¹. Please note that the image should be less than 200 KB and have dimensions of 364x180 pixels for the best results¹.

Source: Conversation with Bing, 11/11/2023
(1) win 10 toast notification with an image in python. https://stackoverflow.com/questions/62826794/win-10-toast-notification-with-an-image-in-python.
(2) winsdk-toast · PyPI. https://pypi.org/project/winsdk-toast/.
(3) Quickstart: App notifications in the Windows App SDK. https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/notifications/app-notifications/app-notifications-quickstart.
(4) ToastNotificationManager Class (Windows.UI.Notifications) - Windows UWP .... https://learn.microsoft.com/en-us/uwp/api/windows.ui.notifications.toastnotificationmanager?view=winrt-22621.
(5) Mo-Dabao/winsdk_toast - GitHub. https://github.com/Mo-Dabao/winsdk_toast.
(6) undefined. https://picsum.photos/48?image=883.
