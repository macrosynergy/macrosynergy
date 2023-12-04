
# FAQs and Troubleshooting

## I am having trouble connecting to DataQuery using the API ...

For the most common issues, such as incorrect credentials, invalid certificates etc, 
the package will raise an exception with a helpful error message.

If you find that the package raises an `HTTPConnection`/`HTTPSConnectionPool` error, 
please check your proxy settings. In scenarios where an error is raised while running 
`check_connection()` (or another download), the error is raised with context to the OAuth 
token request (to "https://authe.jpmchase.com/as/token.oauth2").

You would most likely need to pass your proxy settings to the `JPMaQSDownload` object, as 
shown in the [Connecting via a proxy server](#connecting-via-a-proxy-server) section.
If you are accessing DataQuery from an institutional/enterprise network, please contact 
your IT department to ensure that you have the correct proxy settings.

For organizations using ZScaler - you may have to manually add the ZScaler certificates 
(copy-pasta) to the `certifi` certificate store (typically called `cacert.pem`). You can 
find the location of the `certifi` certificate store by running the following in your Python environment:

```python
import certifi
print(certifi.where())
```

## A function is not working as expected ...

Please check the documentation for the function on our [documentation website](https://docs.macrosynergy.com),
and ensure you are using the latest version of the package.
If you are still having issues, please [raise an issue](https://github.com/macrosynergy/macrosynergy/issues/new/choose) on our GitHub repository.
Please include a minimal reproducible example, and the output of `pip freeze` in your issue.

## I have a feature request ...

Please [raise an issue](https://github.com/macrosynergy/macrosynergy/issues/new/choose), 
and title it "Feature Request: [your feature request]".

## Contributing or creating a pull request ...

Currently, we do not allow a pull request to be created by users outside of the Macrosynergy team.
If you'd like to contribute, please create a fork of the repository, and create a pull request from your fork.

