import winrt.windows.ui.notifications as notifications
import winrt.windows.data.xml.dom as dom

nManager = notifications.ToastNotificationManager
notifier = nManager.create_toast_notifier(r"C:\\path\\to\\your\\app.exe")

tString = """
<toast>
  <visual>
    <binding template='ToastGeneric'>
      <text>Your title</text>
      <text>Your message</text>
      <image placement="appLogoOverride" hint-crop="circle" src="https://url/to/your/image.jpg"/>
    </binding>
  </visual>
</toast>
"""

xDoc = dom.XmlDocument()
xDoc.load_xml(tString)

notification = notifications.ToastNotification(xDoc)

# Display notification
notifier.show(notification)
